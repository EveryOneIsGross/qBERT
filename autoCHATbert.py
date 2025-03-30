"""
It might not be tracking config states correctly between truns f or the teacher to maintain and update.

"""

import warnings
# Must import warnings first and set filters before other imports
warnings.filterwarnings('ignore')

import sys
import torch
from typing import Union, Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import json
import time
import ollama
from pydantic import BaseModel, Field
from puBERT import ParallelBERTGenerator as puBERTGenerator, GenerationConfig as puBERTConfig
from qBERT import ParallelBERTGenerator as qBERTGenerator, GenerationConfig as qBERTConfig, ModelConfig
import re
from colorama import init, Fore, Back, Style
import yaml
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken  # Add to imports
import asyncio
import pickle


class ConversationEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    input_text: str
    output_text: str
    model_name: str
    model_type: str
    config: Dict[str, Any]
    tokens_generated: int
    stream_mode: bool = True
    generation_time: float = 0.0

class SystemUpdate(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    update_type: str
    previous_value: Any
    new_value: Any
    model_type: str

class ModelResponse(BaseModel):
    """Single model response entry"""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_name: str
    model_type: str  # Add to distinguish between 'bert' and 'ollama'
    response: str
    input_text: str  # Add to track what prompted the response
    full_response: Optional[str] = None
    system_prompt: Optional[str] = None
    input_prompt: Optional[str] = None
    reflection_context: Optional[str] = None
    config: Dict[str, Any]
    tokens: int
    generation_time: float = 0.0

class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.response_log = self.log_dir / "responses.jsonl"
        self.system_log = self.log_dir / "system_updates.jsonl"
        self.context_log = self.log_dir / "context.jsonl"
    
    def log_response(self, entry: ModelResponse):
        with open(self.response_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.dict()) + "\n")
    
    def log_system_update(self, update: SystemUpdate):
        """Write system configuration updates"""
        with open(self.system_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(update.dict()) + "\n")
            
    def log_context(self, context_data: Dict[str, Any]):
        """Log full context of each interaction"""
        with open(self.context_log, "a", encoding="utf-8") as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                **context_data
            }
            f.write(json.dumps(entry) + "\n")

class AutoChat(BaseModel):
    pause_duration: float = Field(default=2.0, description="Seconds to wait between exchanges")
    is_paused: bool = Field(default=False, description="Chat pause state")
    max_context_length: int = Field(default=4096, description="Maximum context length to maintain")
    conversation_history: List[str] = Field(default_factory=list)
    max_chunk_size: int = Field(default=512, description="Maximum tokens per chunk for Ollama responses")
    max_bert_context: int = Field(default=256, description="Maximum context tokens for BERT input")
    max_bert_response: int = Field(default=256, description="Maximum tokens for BERT response")

class Reflection(BaseModel):
    """Stores model reflections with vector embeddings"""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    input_text: str
    reflection_text: str
    response_text: str
    model_name: str
    embedding: List[float]
    keywords: List[str]

class VectorIndex:
    """Manages vector storage and search for reflections"""
    def __init__(self, dimension: int = 1024, save_path: str = "data/vector_index.pkl"):
        self.dimension = dimension
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if it exists
        if self.save_path.exists():
            with open(self.save_path, 'rb') as f:
                loaded = pickle.load(f)
                self.reflections = loaded.reflections
                self.embeddings = loaded.embeddings
                print(f"{Fore.CYAN}Loaded {len(self.reflections)} reflections from index{Style.RESET_ALL}")
        else:
            self.reflections: List[Reflection] = []
            self.embeddings: List[List[float]] = []
    
    def add_reflection(self, reflection: Reflection):
        if not reflection.embedding or all(x == 0 for x in reflection.embedding):
            print("Warning: Adding reflection with zero embedding")
        self.reflections.append(reflection)
        self.embeddings.append(reflection.embedding)
        
        # Save after each addition
        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Reflection]:
        if not self.embeddings:
            return []
        
        similarities = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(self.embeddings)
        )[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.reflections[i] for i in top_indices]

class OllamaChat:
    """Manages Ollama's conversation history"""
    def __init__(self, max_history: int = 10, embed_model: str = 'nomic-embed-text', 
                 history_path: str = "data/chat_history.pkl"):
        self.max_history = max_history
        self.model_name = 'qwq'
        self.client = ollama.Client()
        self.embed_model = embed_model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load prompts first
        config_path = Path("config/prompts.yaml")
        with open(config_path) as f:
            self.prompts = yaml.safe_load(f)
        
        # Initialize messages with system prompt
        if self.history_path.exists():
            with open(self.history_path, 'rb') as f:
                self.messages = pickle.load(f)
                print(f"{Fore.CYAN}Loaded {len(self.messages)-1} messages from history{Style.RESET_ALL}")
        else:
            self.messages = [{
                "role": "system",
                "content": self.prompts["ollama_system_prompt"]
            }]
            
        # Set reflection prompts after loading prompts
        self.reflection_prompt = self.prompts["reflection_prompt"]
        self.response_prompt = self.prompts["response_prompt"]
        
        # Initialize vector index last
        self.vector_index = VectorIndex()
        self.logger = Logger()  # Add logger instance
    
    def add_message(self, role: str, content: str):
        """Add message and maintain history limit"""
        self.messages.append({"role": role, "content": content})
        
        # Keep system prompt plus max_history messages
        if len(self.messages) > self.max_history + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]
            
        # Save after each addition
        with open(self.history_path, 'wb') as f:
            pickle.dump(self.messages, f)

    def build_context_messages(self, bert_response: str, reflections: List[Reflection], conversation_history: str) -> List[Dict[str, str]]:
        """Build message list with system prompt, reflections, and current exchange"""
        reflection_context = "\n".join([
            f"Previous reflection: {r.reflection_text}" 
            for r in reflections
        ])
        
        # Format response prompt with context
        formatted_prompt = self.response_prompt.format(
            reflection_context=reflection_context,
            conversation_history=conversation_history,
            bert_response=bert_response
        )
        
        return [
            {"role": "system", "content": self.prompts["ollama_system_prompt"]},
            {"role": "user", "content": formatted_prompt}
        ]

    async def generate_reflection(self, bert_response: str) -> Reflection:
        """Generate reflection using separate prompt"""
        relevant_reflections = self.vector_index.search(await self.generate_embedding(bert_response))
        reflection_context = "\n".join([r.reflection_text for r in relevant_reflections])
        
        formatted_prompt = self.reflection_prompt.format(
            bert_response=bert_response,
            reflection_context=reflection_context
        )
        
        reflection_messages = [
            {"role": "system", "content": self.prompts["ollama_system_prompt"]},
            {"role": "user", "content": formatted_prompt}
        ]
        
        # Log reflection context
        self.logger.log_context({
            "type": "reflection",
            "bert_response": bert_response,
            "relevant_reflections": [r.dict() for r in relevant_reflections],
            "formatted_prompt": formatted_prompt,
            "messages": reflection_messages
        })
        
        reflection = self.client.chat(
            model=self.model_name,
            messages=reflection_messages
        )
        
        reflection_text = reflection['message']['content']
        embedding = await self.generate_embedding(reflection_text)
        keywords = self.extract_keywords(reflection_text)
        
        return Reflection(
            input_text=bert_response,
            reflection_text=reflection_text,
            response_text="",
            model_name=self.model_name,
            embedding=embedding,
            keywords=keywords
        )

    async def generate_response(self, bert_response: str, relevant_reflections: List[Reflection]) -> str:
        """Generate response with context from similar past reflections"""
        reflection_context = "\n".join([f"Previous reflection: {r.reflection_text}" 
                           for r in relevant_reflections])
        
        conversation_history = "\n".join(self.messages[-4:])
        
        formatted_prompt = self.response_prompt.format(
            reflection_context=reflection_context,
            conversation_history=conversation_history,
            bert_response=bert_response
        )
        
        response_messages = [
            {"role": "system", "content": self.prompts["ollama_system_prompt"]},
            {"role": "user", "content": formatted_prompt}
        ]
        
        # Log response context
        self.logger.log_context({
            "type": "response",
            "bert_response": bert_response,
            "relevant_reflections": [r.dict() for r in relevant_reflections],
            "conversation_history": conversation_history,
            "formatted_prompt": formatted_prompt,
            "messages": response_messages
        })
        
        response = self.client.chat(
            model=self.model_name,
            messages=response_messages
        )
        
        response_text = truncate_context(response['message']['content'], 256)
        
        # Log final response
        self.logger.log_context({
            "type": "final_response",
            "raw_response": response['message']['content'],
            "truncated_response": response_text
        })
        
        return response_text

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key terms using tiktoken frequency analysis"""
        # Tokenize and decode back to words
        tokens = self.tokenizer.encode(text.lower())
        words = [self.tokenizer.decode([token]) for token in tokens]
        
        # Filter out common tokens and single characters
        filtered_words = [word.strip() for word in words 
                         if len(word.strip()) > 1 and not word.strip().isnumeric()]
        
        # Get frequency distribution
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top k keywords by frequency
        return sorted(word_freq, key=word_freq.get, reverse=True)[:top_k]

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama's embedding model"""
        #print(f"\n{Fore.CYAN}Generating Embedding{Style.RESET_ALL}")
        #print(f"{Fore.YELLOW}Input:{Style.RESET_ALL}\n{text[:100]}...")
        
        try:
            response = self.client.embed(
                model=self.embed_model,
                input=text
            )
            # Handle Ollama's response structure directly
            if hasattr(response, 'embedding'):
                return response.embedding
            elif hasattr(response, 'embeddings'):
                return response.embeddings[0]
            elif isinstance(response, dict):
                if 'embedding' in response:
                    return response['embedding']
                elif 'embeddings' in response:
                    return response['embeddings'][0]
            
            print(f"Unexpected response structure: {type(response)}")
            # Inspect response structure
            if hasattr(response, '__dict__'):
                print(f"Available attributes: {dir(response)}")
            return [0.0] * self.vector_index.dimension
            
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            print(f"Response type: {type(response)}")
            return [0.0] * self.vector_index.dimension

    async def get_relevant_reflections(self, text: str) -> List[Reflection]:
        """Get relevant reflections for current context"""
        embedding = await self.generate_embedding(text)
        return self.vector_index.search(embedding)

class TeacherFeedback(BaseModel):
    """Structured feedback from teacher agent for hyperparameter tuning"""
    base_temperature: float = Field(ge=0.1, le=2.0, description="Controls randomness in generation")
    min_threshold: float = Field(ge=0.1, le=1.0, description="Token filtering threshold")
    top_k: int = Field(ge=1, le=100, description="Number of candidate tokens")
    context_window: int = Field(ge=16, le=512, description="Context window size")
    notes: str = Field(default="", description="Explanation of adjustments")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in recommendations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "base_temperature": 0.7,
                "min_threshold": 0.5,
                "top_k": 32,
                "context_window": 256,
                "notes": "Adjusted for better coherence",
                "confidence": 0.8
            }
        }

# Add these new models for tracking history
class ParameterUpdate(BaseModel):
    """Record of a single parameter update"""
    parameter: str
    old_value: Any
    new_value: Any
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class TeacherHistory(BaseModel):
    """Historical context of teacher suggestions and parameter changes"""
    suggestion: TeacherFeedback
    applied_updates: List[ParameterUpdate]
    configuration: Dict[str, Any]  # Renamed from model_config
    generation_quality: Dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class TeacherAgent:
    """Evaluates BERT output and suggests hyperparameter adjustments"""
    
    def __init__(self, 
                 ollama_client: Any,
                 model_name: str = "mistral:latest",
                 min_confidence: float = 0.6,
                 logger: Optional[Logger] = None,
                 history_size: int = 5):
        self.client = ollama_client
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.logger = logger or Logger()
        self.history_size = history_size
        self.history: List[TeacherHistory] = []
        
        # Load teacher prompt template with fallback
        try:
            with open("config/prompts.yaml") as f:
                prompts = yaml.safe_load(f)
                self.evaluation_prompt = prompts.get("teacher_evaluation_prompt")
                if not self.evaluation_prompt:
                    raise ValueError("Teacher prompt not found in prompts.yaml")
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not load teacher prompt: {e}{Style.RESET_ALL}")
            self.evaluation_prompt = """
                Analyze this text and suggest parameter updates as JSON:
                Generated text: {generated_text}
                Context: {context}
                Return valid JSON with: base_temperature (0.1-2.0), min_threshold (0.1-1.0),
                top_k (1-100), context_window (16-512), notes (str), confidence (0-1)
            """
    
    def _format_history_context(self) -> str:
        """Format historical context for the prompt"""
        if not self.history:
            return "No previous adjustments."
            
        context = ["Previous adjustments and their effects:"]
        
        for entry in self.history[-self.history_size:]:
            updates = [f"{u.parameter}: {u.old_value} → {u.new_value}" 
                      for u in entry.applied_updates]
            
            context.append(f"\nTimestamp: {entry.timestamp}")
            context.append(f"Suggestion: {entry.suggestion.notes}")
            context.append(f"Changes: {', '.join(updates) if updates else 'No changes applied'}")
            if entry.generation_quality:
                metrics = [f"{k}: {v:.2f}" for k, v in entry.generation_quality.items()]
                context.append(f"Quality Metrics: {', '.join(metrics)}")
            
        return "\n".join(context)
    
    async def evaluate_and_update(self, 
                                generator: Union[puBERTGenerator, qBERTGenerator],
                                bert_response: str,
                                context: str,
                                model_type: str) -> Tuple[TeacherFeedback, bool]:
        """Evaluate output and update model parameters if confidence is high enough"""
        
        current_config = {
            param: getattr(generator.config, param)
            for param in TeacherFeedback.__fields__.keys()
            if hasattr(generator.config, param)
        }
        
        quality_metrics = self._calculate_quality_metrics(bert_response, context)
        
        # Get feedback with historical context
        feedback = await self._get_feedback(
            generated_text=bert_response,
            context=context,
            current_config=current_config,
            history_context=self._format_history_context()
        )
        
        # Log the evaluation - currently only logs the result
        self.logger.log_context({
            "type": "teacher_evaluation",
            "model_type": model_type,
            "bert_response": bert_response,
            "context": context,
            "current_config": current_config,
            "quality_metrics": quality_metrics,
            "feedback": feedback.dict()
        })
        
        # Track applied updates
        applied_updates: List[ParameterUpdate] = []
        updated = False
        
        # Only update if confidence exceeds threshold
        if feedback.confidence >= self.min_confidence:
            updated = await self._update_parameters(
                generator, feedback, model_type, applied_updates
            )
        
        # Record this evaluation in history
        history_entry = TeacherHistory(
            suggestion=feedback,
            applied_updates=applied_updates,
            configuration=current_config,  # Updated to match new field name
            generation_quality=quality_metrics
        )
        self.history.append(history_entry)
        
        # Maintain history size
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
        
        return feedback, updated
    
    def _calculate_quality_metrics(self, text: str, context: str) -> Dict[str, float]:
        """Calculate basic quality metrics for the generated text"""
        try:
            # Length ratio (generated/context)
            length_ratio = len(text.split()) / max(len(context.split()), 1)
            
            # Repetition score (lower is better)
            words = text.lower().split()
            unique_ratio = len(set(words)) / max(len(words), 1)
            
            # Punctuation ratio
            punct_ratio = len([c for c in text if c in '.,!?;:']) / max(len(text), 1)
            
            return {
                "length_ratio": length_ratio,
                "unique_ratio": unique_ratio,
                "punct_ratio": punct_ratio
            }
        except Exception as e:
            print(f"{Fore.RED}Error calculating metrics: {e}{Style.RESET_ALL}")
            return {}
    
    async def _get_feedback(self, 
                           generated_text: str, 
                           context: str,
                           current_config: Dict[str, Any],
                           history_context: str) -> TeacherFeedback:
        """Get structured feedback on generation quality"""
        
        prompt = self.evaluation_prompt.format(
            generated_text=generated_text,
            context=context,
            current_config=json.dumps(current_config, indent=2),
            history_context=history_context
        )
        
        try:
            # Log input prompt using existing structure
            self.logger.log_context({
                "type": "teacher_prompt",
                "prompt": prompt,
                "current_config": current_config,
                "history_context": history_context
            })

            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    "role": "system",
                    "content": "You are a teaching assistant. Respond only with valid JSON."
                }, {
                    "role": "user",
                    "content": prompt
                }]
            )
            
            content = response['message']['content']
            json_str = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_str:
                raise ValueError("No JSON found in response")
                
            feedback_dict = json.loads(json_str.group())
            feedback = TeacherFeedback(**feedback_dict)
            
            # Log response using the TeacherFeedback object
            self.logger.log_context({
                "type": "teacher_response",
                "raw_response": content,
                "feedback": feedback.dict(),
                "model_name": self.model_name
            })
            
            return feedback
            
        except Exception as e:
            print(f"{Fore.RED}Teacher evaluation failed: {str(e)}{Style.RESET_ALL}")
            return TeacherFeedback(
                base_temperature=0.7,
                min_threshold=0.5,
                top_k=32,
                context_window=256,
                notes="Failed to get feedback, using defaults",
                confidence=0.0
            )
    
    async def _update_parameters(self,
                               generator: Union[puBERTGenerator, qBERTGenerator],
                               feedback: TeacherFeedback,
                               model_type: str,
                               applied_updates: List[ParameterUpdate]) -> bool:
        """Apply parameter updates to the generator"""
        try:
            updates = feedback.dict(exclude={'notes', 'confidence'})
            for param, value in updates.items():
                try:
                    old_value = getattr(generator.config, param)
                    generator = update_config(generator, param, str(value))
                    
                    # Record the update
                    applied_updates.append(ParameterUpdate(
                        parameter=param,
                        old_value=old_value,
                        new_value=value
                    ))
                    
                    # Log the update
                    self.logger.log_system_update(SystemUpdate(
                        update_type=param,
                        previous_value=old_value,
                        new_value=value,
                        model_type=model_type
                    ))
                    
                    print(f"{Fore.YELLOW}Updated {param}: {old_value} → {value}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Failed to update {param}: {e}{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Parameter update failed: {str(e)}{Style.RESET_ALL}")
            return False

def create_generator(model_type: str = "qbert", 
                    config: Optional[Union[puBERTConfig, qBERTConfig]] = None,
                    model_config: Optional[ModelConfig] = None):
    """Initialize the model with provided or default configs"""
    if model_type.lower() == "pubert":
        if config is None:
            config = puBERTConfig(
                max_length=1000,
                batch_size=4,
                num_candidates=128,
                embedding_dim=768,
                context_window=32,
                base_temperature=0.5,
                min_threshold=0.8,
                top_k=50,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return puBERTGenerator(config, model_config=model_config or ModelConfig())
    elif model_type.lower() == "qbert":
        if config is None:
            config = qBERTConfig(
                max_length=512,
                batch_size=8,
                num_candidates=128,
                embedding_dim=768,
                context_window=256,
                base_temperature=0.7,
                min_threshold=0.5,
                top_k=32,
                compression_ratio=0.2,
                max_cache_size=16,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return qBERTGenerator(config, model_config=model_config or ModelConfig())
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def update_config(generator: Union[puBERTGenerator, qBERTGenerator], param: str, value: str):
    """Update configuration parameter with type checking"""
    current_config = generator.config
    config_dict = vars(current_config)
    
    try:
        # Type conversion based on current parameter type
        current_type = type(config_dict[param])
        if current_type == bool:
            value = value.lower() == 'true'
        else:
            value = current_type(value)
        
        # Create new config with updated parameter
        new_config_dict = vars(current_config).copy()
        new_config_dict[param] = value
        
        # Determine model type and create appropriate config
        if isinstance(generator, puBERTGenerator):
            new_config = puBERTConfig(**new_config_dict)
            return create_generator("pubert", new_config)
        else:
            new_config = qBERTConfig(**new_config_dict)
            return create_generator("qbert", new_config)
        
    except ValueError:
        raise ValueError(f"Invalid value for {param}. Expected type: {current_type.__name__}")
    except KeyError:
        raise KeyError(f"Unknown parameter: {param}")

def create_ollama_client(model_name: str = "deepseek-r1:8b"):
    """Initialize Ollama client with specified model"""
    return {"model": model_name, "client": ollama}

def truncate_context(text: str, max_tokens: int) -> str:
    """Truncate text to max tokens, preserving most recent complete sentences"""
    # Extract only content between response tags
    response_match = re.search(r'(?<=<response>).*?(?=</response>)', text, re.DOTALL)
    if response_match:
        text = response_match.group().strip()
    else:
        # Fallback: Get last paragraph if no response tags
        paragraphs = text.split('\n\n')
        text = paragraphs[-1].strip() if paragraphs else text
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    tokens = 0
    result = []
    
    for sentence in reversed(sentences):
        sentence_tokens = len(re.findall(r'\w+|[^\w\s]', sentence))
        if tokens + sentence_tokens > max_tokens:
            break
        result.insert(0, sentence)
        tokens += sentence_tokens
    
    return ' '.join(result)

def process_ollama_output(text: str, max_tokens: int = 256) -> str:
    """Process Ollama output to get input for BERT
    1. Try to extract last complete XML tag in final paragraph
    2. If no tags, get last 4 sentences from final paragraphs
    3. Truncate to max tokens preserving start and end
    """
    # First try to find last complete XML tag in final paragraph
    paragraphs = text.split('\n\n')
    last_para = paragraphs[-1]
    
    # Look for answer tags first, then any complete XML tags
    answer_match = re.search(r'<response>(.*?)</response>', last_para, re.DOTALL)
    if answer_match:
        text = answer_match.group(1).strip()
    else:
        # Get last 4 sentences from final paragraphs
        all_sentences = []
        for para in reversed(paragraphs):
            para_sentences = [s.strip() for s in re.split(r'([.!?])\s+', para) if s.strip()]
            for i in range(0, len(para_sentences)-1, 2):
                if i+1 < len(para_sentences):
                    all_sentences.append(para_sentences[i] + para_sentences[i+1])
            if len(all_sentences) >= 4:
                break
        
        text = ' '.join(all_sentences[-4:]).strip()
    
    # If text is too long, preserve start and end
    tokens = re.findall(r'\w+|[^\w\s]', text)
    if len(tokens) > max_tokens:
        keep_tokens = max_tokens // 2
        start_text = ' '.join(tokens[:keep_tokens])
        end_text = ' '.join(tokens[-keep_tokens:])
        text = f"{start_text} ... {end_text}"
    
    return text

def format_thinking(text: str) -> str:
    """Extract and format thinking content"""
    match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    return match.group(1).strip() if match else ""

async def process_exchange(bert_response: str, ollama_chat: OllamaChat):
    """Process a single exchange with reflection and vector search"""
    # Generate reflection
    reflection = await ollama_chat.generate_reflection(bert_response)
    
    # Search for relevant past reflections
    query_embedding = reflection.embedding
    relevant_reflections = ollama_chat.vector_index.search(query_embedding)
    
    # Generate response using relevant reflections
    response = await ollama_chat.generate_response(bert_response, relevant_reflections)
    
    # Update reflection with final response
    reflection.response_text = response
    
    # Add to vector index
    ollama_chat.vector_index.add_reflection(reflection)
    
    return response

async def main():
    """Initialize models and run autonomous chat interface"""
    num_tokens = 64
    stream_mode = True
    model_type = "qbert"
    
    logger = Logger()
    generator = create_generator(model_type)
    auto_chat = AutoChat()
    ollama_chat = OllamaChat(max_history=4)
    
    init(autoreset=True)
    
    banner = f"""
    {Fore.CYAN}╔═══════════════════════════════════════╗
    ║         {Style.BRIGHT}Autonomous BERT Chat{Style.NORMAL}          ║
    ╚═══════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(banner)
    
    # Get seed input from user
    seed_input = input(f"{Fore.GREEN}> {Style.RESET_ALL}")
    auto_chat.conversation_history.append(seed_input)
    
    print(f"\n{Fore.YELLOW}Press Ctrl+C to pause/resume, Ctrl+D to quit{Style.RESET_ALL}")
    
    while True:
        try:
            if not auto_chat.is_paused:
                # 1. BERT Response
                bert_input = truncate_context(auto_chat.conversation_history[-1], auto_chat.max_bert_context)
                print(f"{Fore.BLUE}{model_type.upper()}: ", end="", flush=True)
                bert_response = ""
                start_time = time.time()
                
                for token in generator.generate_stream(
                    f"{bert_input}\n",
                    num_tokens=min(num_tokens, auto_chat.max_bert_response)
                ):
                    print(f"{token}", end="", flush=True)
                    bert_response += token
                
                generation_time = time.time() - start_time
                
                # Log BERT response
                logger.log_response(ModelResponse(
                    model_name=model_type,
                    model_type='bert',
                    response=bert_response,
                    input_text=bert_input,
                    config=vars(generator.config),
                    tokens=len(re.findall(r'\w+', bert_response)),
                    generation_time=generation_time
                ))
                
                print(Style.RESET_ALL)
                
                # Add teacher evaluation here
                teacher = TeacherAgent(
                    ollama_client=ollama_chat.client,
                    model_name=ollama_chat.model_name,
                    logger=logger
                )
                
                feedback, updated = await teacher.evaluate_and_update(
                    generator=generator,
                    bert_response=bert_response,
                    context="\n".join(auto_chat.conversation_history[-4:]),
                    model_type=model_type
                )
                
                if updated:
                    print(f"\n{Fore.CYAN}Teacher Notes: {feedback.notes}{Style.RESET_ALL}")
                
                # 2. Generate Reflection & Embedding

                reflection = await ollama_chat.generate_reflection(bert_response)
                
                # 3. Search Vector Index for Similar Reflections

                similar_reflections = ollama_chat.vector_index.search(reflection.embedding)
                print(f"Found {len(similar_reflections)} relevant reflections")
                """
                # Print out found reflections
                if similar_reflections:
                    print(f"\n{Fore.YELLOW}Previous Reflections:{Style.RESET_ALL}")
                    for i, ref in enumerate(similar_reflections, 1):
                        #print(f"\n{Fore.CYAN}Memory {i}:{Style.RESET_ALL}")
                        #print(f"Input: {ref.input_text[:100]}...")
                        #print(f"Reflection: {ref.reflection_text[:200]}...")
                        #print(f"Response: {ref.response_text[:100]}...")
                        print(f"Timestamp: {ref.timestamp}")
                else:
                    print(f"\n{Fore.YELLOW}No previous reflections found{Style.RESET_ALL}")
                """
                # 4. Build Full Context with Reflections
                #print(f"\n{Fore.CYAN}{'='*50}")
                #print(f"Stage 4: Building Context{Style.RESET_ALL}")
                #print(f"Including {len(similar_reflections)} previous reflections")
                messages = ollama_chat.build_context_messages(
                    bert_response=bert_response,
                    reflections=similar_reflections,
                    conversation_history="\n".join(auto_chat.conversation_history[-4:])
                )
                
                # 5. Get Ollama Response
                print(f"{Fore.MAGENTA}{ollama_chat.model_name}: ", end="", flush=True)
                raw_response = ""
                
                # Stream response directly
                for chunk in ollama_chat.client.chat(
                    model=ollama_chat.model_name,
                    messages=messages,
                    stream=True
                ):
                    if chunk['message']['content']:
                        print(f"{chunk['message']['content']}", end="", flush=True)
                        raw_response += chunk['message']['content']
                print(Style.RESET_ALL)
                
                # Extract clean response for history
                clean_response = truncate_context(raw_response, 256)
                
                # 6. Update Memory & History
                reflection.response_text = raw_response  # Store full response with thinking
                ollama_chat.vector_index.add_reflection(reflection)
                ollama_chat.add_message("user", bert_response)
                ollama_chat.add_message("assistant", clean_response)  # Store clean response
                auto_chat.conversation_history.extend([bert_response, clean_response])
                
                # 7. Log Responses
                logger.log_response(ModelResponse(
                    model_name=ollama_chat.model_name,
                    model_type="ollama",
                    response=clean_response,
                    input_text=messages[-1]["content"],
                    full_response=raw_response,  # Keep full response in logs
                    system_prompt=messages[0]["content"],
                    input_prompt=messages[-1]["content"],
                    reflection_context="\n".join(f"Previous reflection: {r.reflection_text}" 
                                               for r in similar_reflections),
                    config={"model": ollama_chat.model_name},
                    tokens=len(re.findall(r'\w+', clean_response)),
                    generation_time=time.time() - generation_time
                ))
                
                # 8. Maintain Context Length
                if len(auto_chat.conversation_history) > auto_chat.max_context_length:
                    auto_chat.conversation_history = auto_chat.conversation_history[-auto_chat.max_context_length:]
                
                time.sleep(auto_chat.pause_duration)
                
            else:
                # Handle commands while paused
                user_input = input(f"\n{Fore.YELLOW}Chat paused. Enter command or press Enter to resume: {Style.RESET_ALL}")
                if not user_input:
                    auto_chat.is_paused = False
                    print("Resuming chat...")
                    continue
                    
                if user_input.lower() == '/quit':
                    print(f"{Fore.RED}Goodbye!{Style.RESET_ALL}")
                    break
                
                # Handle existing commands...
                if user_input.startswith('/'):
                    parts = user_input[1:].split(maxsplit=1)
                    command = parts[0].lower()
                    
                    # Direct parameter update
                    if hasattr(generator.config, command):
                        try:
                            if len(parts) != 2:
                                print(f"Please provide a value for {command}")
                                continue
                            old_config = vars(generator.config).copy()
                            generator = update_config(generator, command, parts[1])
                            logger.log_system_update(SystemUpdate(
                                update_type="config",
                                previous_value=old_config[command],
                                new_value=parts[1],
                                model_type=model_type
                            ))
                            print(f"Updated {command} to {parts[1]}")
                            continue
                        except (ValueError, KeyError) as e:
                            print(f"Error: {str(e)}")
                            continue
                
                if user_input.lower() == '/help':
                    print("\nCurrent Configuration:")
                    print(f"BERT Model Type: {model_type}")
                    print(f"Ollama Model: {ollama_chat.model_name}")
                    for key, value in vars(generator.config).items():
                        print(f"{key}: {value}")
                    print("\nCommands:")
                    print("/model <pubert|qbert> - Switch BERT model")
                    print("/ollama <model> - Switch Ollama model")
                    print("/models - List available Ollama models")
                    print("/device <cpu|cuda> - Switch device")
                    print("/tokens <number> - Set token count")
                    print("/stream - Toggle streaming mode")
                    print("/clear - Clear screen")
                    print("/quit - Exit program")
                    print("\nModel Configuration Commands:")
                    print("/bert_model <name> - Change BERT model (e.g. bert-base-uncased)")
                    print("/sentence_model <name> - Change sentence transformer model")
                    print("/max_length <int> - Set maximum sequence length")
                    print("/batch_size <int> - Set batch size")
                    print("/num_candidates <int> - Set number of candidates")
                    print("/embedding_dim <int> - Set embedding dimension")
                    print("/context_window <int> - Set context window size")
                    print("/base_temperature <float> - Set base temperature")
                    print("/min_threshold <float> - Set minimum threshold")
                    print("/top_k <int> - Set top-k value")
                    if model_type == "qbert":
                        print("/compression_ratio <float> - Set compression ratio")
                        print("/max_cache_size <int> - Set maximum cache size")
                    continue

                if user_input.lower() == '/models':
                    try:
                        models = ollama_chat.client.list()
                        print("\nAvailable Ollama Models:")
                        if hasattr(models, 'models'):
                            for model in models.models:
                                if hasattr(model, 'model'):
                                    print(f"- {model.model}")
                                else:
                                    print(f"- {str(model)}")
                        else:
                            print("No models found")
                    except Exception as e:
                        print(f"Error listing models: {str(e)}")
                        print("Make sure Ollama server is running")
                        if hasattr(models, 'models'):
                            print("Debug - First model attributes:", dir(models.models[0]))
                    continue

                if user_input.lower().startswith('/ollama '):
                    try:
                        _, new_model = user_input.split()
                        old_model = ollama_chat.model_name
                        ollama_chat.model_name = new_model
                        logger.log_system_update(SystemUpdate(
                            update_type="ollama_model",
                            previous_value=old_model,
                            new_value=new_model,
                            model_type="ollama"
                        ))
                        print(f"Switched Ollama model to {new_model}")
                    except Exception as e:
                        print(f"Error switching Ollama model: {str(e)}")
                    continue
                
                if user_input.lower().startswith('/device '):
                    try:
                        _, device = user_input.split()
                        if device not in ['cpu', 'cuda']:
                            raise ValueError("Device must be 'cpu' or 'cuda'")
                        if device == 'cuda' and not torch.cuda.is_available():
                            print("CUDA is not available on this system")
                            continue
                        old_device = generator.config.device
                        generator = update_config(generator, 'device', device)
                        logger.log_system_update(SystemUpdate(
                            update_type="device",
                            previous_value=old_device,
                            new_value=device,
                            model_type=model_type
                        ))
                        print(f"Switched to {device.upper()}")
                    except Exception as e:
                        print(f"Error switching device: {str(e)}")
                    continue
                
                if user_input.lower().startswith('/tokens '):
                    try:
                        _, value = user_input.split()
                        old_tokens = num_tokens
                        num_tokens = int(value)
                        # Also update the max_bert_response limit
                        auto_chat.max_bert_response = num_tokens
                        logger.log_system_update(SystemUpdate(
                            update_type="tokens",
                            previous_value=old_tokens,
                            new_value=num_tokens,
                            model_type=model_type
                        ))
                        print(f"Will generate {num_tokens} tokens")
                    except ValueError:
                        print("Please provide a valid number")
                    continue
                
                if user_input.lower() == '/stream':
                    old_stream = stream_mode
                    stream_mode = not stream_mode
                    logger.log_system_update(SystemUpdate(
                        update_type="stream_mode",
                        previous_value=old_stream,
                        new_value=stream_mode,
                        model_type=model_type
                    ))
                    print(f"Streaming mode: {'ON' if stream_mode else 'OFF'}")
                    continue
                
                if user_input.lower().startswith('/model '):
                    try:
                        _, new_model_type = user_input.split()
                        if new_model_type.lower() not in ['pubert', 'qbert']:
                            raise ValueError("Model type must be 'puBERT' or 'qBERT'")
                        old_model = model_type
                        model_type = new_model_type.lower()  # Store lowercase version
                        generator = create_generator(model_type)
                        logger.log_system_update(SystemUpdate(
                            update_type="model_type",
                            previous_value=old_model,
                            new_value=model_type,
                            model_type=model_type
                        ))
                        print(f"Switched to {model_type}")
                    except Exception as e:
                        print(f"Error switching model: {str(e)}")
                    continue
                
                if user_input.lower().startswith('/bert_model '):
                    try:
                        _, model_name = user_input.split(maxsplit=1)  # Split only on first space
                        model_name = model_name.strip('"')  # Remove any quotes
                        old_model = generator.bert_model.name_or_path
                        # Create new model config with updated model name
                        model_config = ModelConfig(
                            bert_model_name=model_name,
                            tokenizer_name=model_name,  # Usually same as model
                            sentence_transformer_name=generator.model_config.sentence_transformer_name,
                            attn_implementation=generator.model_config.attn_implementation
                        )
                        # Recreate generator with new model config
                        generator = create_generator(
                            model_type=model_type,
                            config=generator.config,  # Keep existing generation config
                            model_config=model_config
                        )
                        logger.log_system_update(SystemUpdate(
                            update_type="bert_model",
                            previous_value=old_model,
                            new_value=model_name,
                            model_type=model_type
                        ))
                        print(f"Switched BERT model to {model_name}")
                    except Exception as e:
                        print(f"Error switching BERT model: {str(e)}")
                    continue
                
                if user_input.lower().startswith('/sentence_model '):
                    try:
                        _, model_name = user_input.split()
                        old_model = generator.model_config.sentence_transformer_name  # Access from model_config
                        model_config = ModelConfig(
                            bert_model_name=generator.bert_model.name_or_path,
                            tokenizer_name=generator.tokenizer.name_or_path,
                            sentence_transformer_name=model_name
                        )
                        generator = create_generator(
                            model_type=model_type,
                            config=generator.config,
                            model_config=model_config
                        )
                        logger.log_system_update(SystemUpdate(
                            update_type="sentence_model",
                            previous_value=old_model,
                            new_value=model_name,
                            model_type=model_type
                        ))
                        print(f"Switched sentence transformer to {model_name}")
                    except Exception as e:
                        print(f"Error switching sentence transformer: {str(e)}")
                    continue
                
        except KeyboardInterrupt:
            auto_chat.is_paused = not auto_chat.is_paused
            print(f"\n{Fore.YELLOW}{'Chat paused. Press Ctrl+C again to resume.' if auto_chat.is_paused else 'Resuming chat...'}{Style.RESET_ALL}")
            
        except EOFError:
            print(f"\n{Fore.RED}Goodbye!{Style.RESET_ALL}")
            break
            
        except Exception as e:
            print(f"\n{Fore.RED}Unexpected error: {str(e)}{Style.RESET_ALL}")
            auto_chat.is_paused = True

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1) 