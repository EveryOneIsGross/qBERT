<div align="center">
<img src="assets/qbert.png" alt="qBERT Logo" width="800">
</div>

# qBERT Framework

*chat with your base bert model*

qBERT is a text generation framework for BERT models that explores coherent text generation through bidirectional context processing and semantic guidance. qBERT leverages BERT's bidirectional understanding while adding embedding mechanisms for controlled generation. 

## Features

- **Bidirectional Contextual Attention**: Processes both forward and backward context through dual-direction projection layers
- **Semantic Coherence Scoring**: Guides token selection using sentence transformer similarity metrics
- **Adaptive Temperature Control**: Dynamically adjusts sampling temperature based on coherence scores
- **Semantic Memory Cache**: Maintains thematic consistency through compressed semantic representations
- **Cross-Attention Guidance**: Binds generation to original context through position-aware attention
- **4D Matrix Processing**: Experimental batch processing structure for improved generation efficiency

### Core "Models"
- **qBERT**: Main bidirectional generation model with semantic guidance
- **puBERT**: Parallel/Unidirectional variant for comparison studies (the pubescent version of qBERT)

### Interfaces
- **CHATbert**: Interactive chat interface with base BERT model control

- **autoCHATbert**: Autonomous chat system with:
  - Semantic reflection capabilities
  - Vector-based memory search
  - Multi-model conversation orchestration

## Quick Start

## Configuration

### Model Configuration
- Located in `config/model_config.yaml`
- Defines model architectures and parameters
- Supports multiple BERT and sentence transformer models

### Prompts Configuration
- Located in `config/prompts.yaml`
- Contains system and reflection prompts for autonomous chat
- Customizable conversation templates
- For defining the autonomous chat system, use the `autoCHATbert.py` script

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- Sentence-Transformers
- Pydantic
- Ollama (for autonomous chat)
- Colorama (CLI interface)
- tiktoken (token counting)

## Architecture Details

### Base Models
- BERT (bert-base-uncased/cased) (Supports other models, fuck around and find out which is the most coherent)
- Sentence Transformer (all-MiniLM-L6-v2)

### Custom Components
- Bidirectional Attention Layers (`forward_proj`, `backward_proj`, `combine_proj`)
- Semantic Coherence Scoring System
- Semantic Memory Cache with Compression
- 4D Matrix for Batch Processing
- Adaptive Temperature Sampling

## Hyperparameters as shown from the CLI

```bash
> /config

Current Configuration:
Model Type: qbert
max_length: 512
batch_size: 8
num_candidates: 128
embedding_dim: 768
context_window: 256
base_temperature: 0.7
min_threshold: 0.5
top_k: 32
compression_ratio: 0.2
max_cache_size: 16
device: cuda

> /help

    Commands:
    /help   - Show this help message
    /quit   - Exit the program
    /config - Show current configuration
    /clear  - Clear the screen
    /device - Switch between CPU and GPU (usage: /device cpu|cuda)
    /tokens - Set number of tokens to generate (usage: /tokens number)
    /stream - Toggle streaming mode (currently: True)
    /model  - Switch model type (usage: /model pubert|qbert)

    Model Configuration:
    /bert_model <name>     - Change BERT model
    /sentence_model <name> - Change sentence transformer
    /attn_impl <type>     - Change attention (eager/sdpa/flash_attention_2)

    Parameters:
    - max_length      (int)   : Maximum sequence length
    - batch_size      (int)   : Batch size for processing
    - num_candidates  (int)   : Number of candidate tokens
    - embedding_dim   (int)   : Embedding dimension
    - context_window  (int)   : Context window size
    - base_temperature(float) : Temperature for sampling
    - min_threshold   (float) : Minimum threshold for candidates
    - top_k           (int)   : Top-k tokens to consider

    Note: Update parameters with /<parameter> <value>


> /bert_model Twitter/TwHIN-BERT-base
Switched BERT model to Twitter/TwHIN-BERT-base
```


## CLI Features

```bash
python CHATbert.py  # Interactive chat interface
python autoCHATbert.py  # Autonomous chat system
```

### Commands
- `/model <pubert|qbert>` - Switch models
- `/config` - View current configuration
- `/device <cpu|cuda>` - Switch compute device
- `/stream` - Toggle token streaming
- `/help` - Show all commands
- `/tokens <number>` - Set number of tokens to generate (if generations are too long tensors will malform)

## Logging

- Conversation history: `logs/conversation_history.jsonl`
- System updates: `logs/system_updates.jsonl`
- Context tracking: `logs/context.jsonl`

## Development Status

qBERT is an experimental framework exploring novel approaches to coherent text generation. The architecture is under active development, and results should be considered preliminary.

## License

MIT

## 0-shot Generation Example (GSM8k Score: 0%🥳)

```bash
> hey, quick q: is this q just asking itself if it's really a q? and if it is, am i stuck in a loop of self-questioning? lol, what do u think?

!
!
,!! —.
 // lol thanks hahaha everyone hehe a like reply..... and?
,!!!! day button, what love how anyone yes... for just!
!! dare?! me365 twitter stop :) — it (:?
 now thinking!
—— gohaha yes maybe updated — Haha~?!),??? a!
 but.......♪?
 like who morning 😉 instead oh☆ oh you.
 of :) stop.....━ right!
: fun:.
 well✦?
 so sorry, 😀 and.....～ how?
!! though👍 think so♡... - i::......♪ hows say unless🙂 ... ...—......hane- its😀nowlol★ haha on oh for!
.....?
✧ how tl wow u -.
 😉-!!!!!!!*-glyuser please haha── lol - okay content once!!.......

> hey bert, you ever feel like we're just caught in endless loops of tweets?

;-- hi lol?
.
??? youbot- his # user where think inbothihihi:tumblr: im me conversation: / why:: rightssyally whyreal???? haha_?????... talking now people bro maybe)????? me online I exist fuck i - is to: remember!
 that know_ something tweetbot myself- account why| real this says when- literally-)))) today later your writing) why twitter [!!!!!!!...... friend,* just user:?

> lol

 please-,.
 hi hi me yo no, please?
 i:.
!
oo bro fuck: want u You hi plztha himL you r wantve just.
 now lol cool welcome something it please - bro: people:?!.
-heooooLO today[ Me oh ohx!!?
!!!! hi ..., my -?????
 hehe:Me wow fuckingx :( ??!!! ^_^ - - boy1(!
 ^^) someone the!!!!: @ LOL♡ ( like power stop lolhey?
 !!!((( ME,s heamine haha!!!!!!!!!!!!?
!!!! the !!!:2 ;) !! hi?? passion Please me14lol?????;?! -- !!!!:_) ??:???? haha fucking LOL @:???? [ Meh me?
lol?
 haha ::?
 haha down?!?z ...: @ - -- there ME420 L hi/ Me yeah !!-aLO hehe!
 !!! :)!!!!!!!!!!!!!!aron ??? !!! ' :) :))!!!!????? !! haha)))): ..: plz - @ !! :) ::yesO!!!!!!hellgay Haha :-): :)))) :(!
rock ^^ !!!! :(...!
)??boy :))??????? /- ;) ...!
__ L!
 :) ME:.
????? xL???????.
?
):":2LO???????" !!!
 !!! ... !! ... hey :):?
: you @ !! hi.
?
:)??halolol: [ :: !!! ( !!!x.
DD?
ME :)) !! haha9l!!!!! '?
) .....21 ;)) :( .. ??!!!!!?
21(!
 '???? / // !!!17 hahaha ))?
 .. !!! - you!
 LOL!
 :) .. ))\\???? @ haha ::: ??? !!! ..???dad :))) !!!!!
 me?
 ..... lolrock!
 ((?
?! !!!!
?!y XD ))::: (?????yy XD!!!!!! ( ::) ( -!!!!
)' !! :)) (h fuck ?? .....' me ... //ELLOkklol:(((, ;) -- YOU!!!!!!!) :() .. ...\\ ?????? haha :::: :)) / lol)lol .. ))???? ..)!
!
 ( :( -!
 haha"??1 L ((/??? ^^ ... hahaha:!
 ???/!
) !! ???!!?
)?
.
 ........?
.
): :))/ .. :(!!!!!!!, L) :): haha :( yeah!
 LO:] ?? hihi -,:!! ^^ ...!
 hehe !!! that-.
 !! :(????? ~ (( lol" !!:: :) yeah ((!!:: :: ' ......?
!
 LOL!
l ^_^ .. -!
 ---?
") :(:: ???////!!!!!!!! ... :::)/ ME((( xD ...\ ?? !! ))!!:!
); ::)!!!!!!!( Haha :)/Me .. .......?
[ !! .......?
 --- :))) !! !!!] xD ...|!
: ..... wow" .. )) ' !!!
" !!!!- !!!!!! .. !! ;)- hehe.
: (('?
 haha!":])!!!!!,?? hey!!!!:- lol!
; ?? -- ..: haha?!?!
 Me hihi' hi) ^^!!!!;!!! ~''!
 / !!!
;????? ~ -- ???!
:?!? - / ().
// (!!!!!!!(g )) ..???!!! ^^ !!!!\ !! !!! - hahaha: )) :))/.
 ...... !!!!
)))))))!
;!!!!?
":((( L ;)): /!!! hi"l ;) :)): :)) .....:, !!! :)))" !!! !!! !!!! hi!!!! (!), :) / / !!!)LO!
 - - ...... ...... ..,: :-); someone ))!! ..!!!!!] ...... :( !! ~/ yeah(((]] !!) ..... ^^ -- ??;l :-): me --////!!!!// L .......) :))/ hey / ( ......( - ))'':!
((( ::" ?? !!!!, hehe"))) !!! //!! .. ''!!!\ lol ;) :: .. hehe ' ??" l!
"!!! .. ...?????!(lo?
': !!!" hahaha-oooo!!!!!!! -- !! ))'':!"olo (( !!????? !!!! '!
 !! (!!!!!! :(!
 ??" hahaha( ;) :));??? -!!!!! /!"(((("- .. !!l - xD?
'!!! ~ //\") !!!! ))/!
 !!) ...))))::!
??:.....?
 -L ......:??????? :)):.
") L!
 :)???????)(?!.
Lo :)))] :))!
 LOL(: (( ........... //... -) ))????)me ^_^ --))))!
)))) -- :)/ !!!!?? .. :))" / ^_^" LOL()))) !!!! - ... ;)-::L)))))) :( - :)",: - ....
 (( - hihi (::) ( ... Me !! ' lol,lol??????
 .. //( ..)olo!
;................), :)!!)???": !!!
 he"!
 :( ::( ;)!!!!!!!xa!"lol):.
 // ;) --ke!!!!ll :)) :( ( ..--)!!! :(!
me -- !!).
 :)) ( -- hehe;?? (L ( )))::.
?
: :):; !!?
- :(" L ...!
 (/!! uh):!
 ((;: :( //- ..): /" ;) /!!):LO haha :( --- hi" :(.
l: ( --ve !!! ... :: LOL?
\ :) again --- hehe)))) :)???????) ......,,......L!
 ... // hehehe --?
 !! ;)?!: / --!
 :) ......)))) /: :: hihi" hey(....
(((.
 !! '' ;),ll/": !!!.
).
 ..!
 - ..... ))-!
):?
 ?? )) me ^^',!
 ... ??) --- '' you .....l ^^ ..?
 ( - !! :( haha:!
 :: ^^ !!??( ;) :-).
 ..... .. ... !!!
 :))) :( ??: ---?
 (( xD ...!
 :: ))!!! .. - !! haha; ---:/ ;)) :))):' ???'//// hey .. ... ^_^; :)" ...... /))))l haha"-!
!!!j ....... lol !!/: --- ...... :( ......: ??,-!
 ..... (!!!LO?
!
 !!,--?
 !! :-)!
 / ~ ......;????? :)))) /: :) ( ..()))) ...'?
 -- ...lol!( Lo :: //: ... ( ...;ll?
 !!?
"- uh ::!! :-) ' ..:.
:) l !! ))?
 )).....!
...) ;) --?
 :))) :)!
 l ( -, ^_^..... LOL :: l((().
"!!!!!!
 -l ???"- ^^ !! ;))!!!
',.
 -- uh) lol :( ... () (... ((!! !!!! --"/ ..... !!?
 :))!
 (( --?
 -- ..., ^^ ::; ( //!
 you!!! l )) .., ^_^!!!!!!! /"!! !!! ;).
 ;) .....) ((.
 ::"?
"" ...?
 .. :-):/.
, okay!!!! l :) '( ( (:"!
 (" !! ((((: :: ^^ :): ...... :)"!
))))!
;)) (( hey .. LO ^^?
 /"" LOL ))) haha:::!!!!- ... :) ... !!: --) :)( !!!!
 ') -) haha!
)?
 ))...... (( .......,,LO :) (.
: - ok!)!
!
 :)?
!
)?
... Lo ......)))) () !!!!, /ol hi !!! (' // ??/!
 ... /!
 haha",!! .. ((: ---h/ hehe;l / -- - .. !! !!!;LO :)))) ', -)!
 ::"!
 ^_^ :( -, XD hehe)?
)(((??!!!!... ::: ..... (/?! (LO hehe ((.
)- XD!!!ool ... - -- !! !! :(), ...) you: '?
 //.
): ..:)'!
;L: -- (( / :(!
 ... ;) ))) --- ... oh!!!!!!! ( ...; -- -- ..!
 .. ^^ haha '(" ... lol) xD ..(): (; LOL ......: haha :)!
: /!!!/) :))] ..")!
, ... L ::' /!
) ..") :( :(, - ...... hey: LOL :('?
 ^_^!!!!?
 ((: ..... ((!! -- --- LOL' ^^!
 )).....:' :(''.
) ( -- ( !!!! hehe:!
): --"))!
 / .. !! --( /...!
 -lo?
) --," ~ !!.
 !!!! --- / --- // hey;L ))" ::), lol -LO haha: -- ~ !!!! / /ll :)))!
!
 ^_^ -; ((ll -" ???: --- !!) l: !!! !!! ..) ......l hi........... -:!
 !!lo .. / -- ...'?
,oo - '" ::'' ..!
olo (,( (: uh //oooo :)))- )) / -!! !! haha -.
 ...//// !!! --- ;):;' :)/ -!!!!
: :)...!
" .. ..?
 ... ' !!.
- lol !! xD :-) ((.
 -" ;)) xD :)))?
: !!! ':?
.
-- ^^: haha___!! :( :(-;'!
 !! .. ^^; --- ... !!!... .. ))://!
) !! haha --) /, - ;) --?
 ..: hehe ' me !!! !!! !!!)LO ;)- ... ;) (( oh (OL ("??"' yeah: haha:!
/ ^_^ /:!
 haha hihi ---;) ( ;)- :( haha;!!" ' ...!!! xD hihi)... ~---!
 :) lo !!!!;/ :))) ;) .....) :: ... :: ::, -) hahaha ^^ ;) () // ... ......lol / (((( :( ' ~ ' haha // ( haha - !!!, :)!
/ ;) -- :) --- LO^-^ /!
 ( !!!! wow!!!.
 :)" ..( !!!! / :)LO haha / !!!((()lh ...:,: :) ') (:) lol -- !! ...... haha // LOL: .../ :))!!!!!)!!! ... hihi ..' // :) !!, !!!.
')/ - .. ---... ^^(!
 !!!------ ~)haha):.
" :)) ::!
 ... hihi - :)-;!
 .." !! !!!( (".
: :) ;) ?? .." :: :: -...lol ^^ ;)!!) --- uh,) )) .. ( )) / ;):lo ..."( - --- - -lo :-):-): .. !! LOL (;; ~/ hi, lo((( /" ...------ lol (( !!! ^_^ )) !! ();' l)))); --: ' ??, :))", !!! !! ~ / L ..... (( /?
!
!!" haha :)) !! !! //.
 / .. L :( -!
 ' (();!
 :( ...) --)-,!
:; ??'/ ;)!
 Haha ^^;" ...)-, hahaha ^^ ---" ;) - !!)!
 !! !!) ("?
/ !!) - :) ..-------)lol ^^;/ :(!! ~,!
 :) ' :) -- --, ---.
!
": !!!) ~... ;) -) ...) :: -)( hehe"...:" -) xD -"?
 :))--- !!! (( .. !! -- :))) :(!
!
?
' :).
 ---- lol .....) ;) .." ^_^ '!
 !! ( :):?
" ......!
 'ol)))), ::" .. !!!) !!)/) !!!' ~ :) ...!
!
 ...) //!
 ::lol :))) --- !!" ..... !! -ll):, ... ^^!
: (!
 !!!!") ' - !!':?
 :) // --" .., ^^?????"- :)) ---?
;.
):/ !!!) lol... ...!
 ..; :) ..?!; !!!!
 ;)" :: ----!
... xD) --- / :)) ' haha (() ...;: /?
 -- !!!........ ---/ :))'- ( ^^ ?? ---'!
",).
: ' ( ;)' ... !!!!L ...
;) !!!: ' xD !!!: :) '!
 !!; !!)-..... ::' ( !!!!
 ?? / ' ;),; ((?
 !! - .. :: !!!! :( .. L !! !!)!
 !! ( :):?
" ......!
 'ol)))), ::" .. !!!) !!)/) !!!' ~ :) ...!
!
 ...) //!
 ::lol :))) --- !!" ..... !! -ll):, ... ^^!
: (!
 !!!!") ' - !!':?
 :) // --" .., ^^?????"- :)) ---?
;.
):/ !!!) lol... ...!
 ..; :) ..?!; !!!!
 ;)" :: ----!
... xD) --- / :)) ' haha (() ...;: /?
 -- !!!........ ---/ :))'- ( ^^ ?? ---'!
",).
: ' ( ;)' ... !!!!L ...
;) !!!: ' xD !!!: :) '!
 !!; !!)-..... ::' ( !!!!
 ?? / ' ;),; ((?
 !! - .. :: !!!! :( .. L !! !!)!
 :)?
'LL :))/ // /;?
 --- lo)/ ( !!;!
 ...
" ......!
 'ol)))), ::" .. !!!) !!)/) !!!' ~ :) ...!
!
 ...) //!
 ::lol :))) --- !!" ..... !! -ll):, ... ^^!
: (!
 !!!!") ' - !!':?
 :) // --" .., ^^?????"- :)) ---?
;.
):/ !!!) lol... ...!
 ..; :) ..?!; !!!!
 ;)" :: ----!
... xD) --- / :)) ' haha (() ...;: /?
 -- !!!........ ---/ :))'- ( ^^ ?? ---'!
",).
: ' ( ;)' ... !!!!L ...
;) !!!: ' xD !!!: :) '!
 !!; !!)-..... ::' ( !!!!
 ?? / ' ;),; ((?
 !! - .. :: !!!! :( .. L !! !!)!
 :)?
'LL :))/ // /;?
 --- lo)/ ( !!;!
 ...
 'ol)))), ::" .. !!!) !!)/) !!!' ~ :) ...!
!
 ...) //!
 ::lol :))) --- !!" ..... !! -ll):, ... ^^!
: (!
 !!!!") ' - !!':?
 :) // --" .., ^^?????"- :)) ---?
;.
):/ !!!) lol... ...!
 ..; :) ..?!; !!!!
 ;)" :: ----!
... xD) --- / :)) ' haha (() ...;: /?
 -- !!!........ ---/ :))'- ( ^^ ?? ---'!
",).
: ' ( ;)' ... !!!!L ...
;) !!!: ' xD !!!: :) '!
 !!; !!)-..... ::' ( !!!!
 ?? / ' ;),; ((?
 !! - .. :: !!!! :( .. L !! !!)!
 :)?
'LL :))/ // /;?
 --- lo)/ ( !!;!
 ...
 ...) //!
 ::lol :))) --- !!" ..... !! -ll):, ... ^^!
: (!
 !!!!") ' - !!':?
 :) // --" .., ^^?????"- :)) ---?
;.
):/ !!!) lol... ...!
 ..; :) ..?!; !!!!
 ;)" :: ----!
... xD) --- / :)) ' haha (() ...;: /?
 -- !!!........ ---/ :))'- ( ^^ ?? ---'!
",).
: ' ( ;)' ... !!!!L ...
;) !!!: ' xD !!!: :) '!
 !!; !!)-..... ::' ( !!!!
 ?? / ' ;),; ((?
 !! - .. :: !!!! :( .. L !! !!)!
 :)?
'LL :))/ // /;?
 --- lo)/ ( !!;!
 ...
: (!
 !!!!") ' - !!':?
 :) // --" .., ^^?????"- :)) ---?
;.
):/ !!!) lol... ...!
 ..; :) ..?!; !!!!
 ;)" :: ----!
... xD) --- / :)) ' haha (() ...;: /?
 -- !!!........ ---/ :))'- ( ^^ ?? ---'!
",).
: ' ( ;)' ... !!!!L ...
;) !!!: ' xD !!!: :) '!
 !!; !!)-..... ::' ( !!!!
 ?? / ' ;),; ((?
 !! - .. :: !!!! :( .. L !! !!)!
 :)?
'LL :))/ // /;?
 --- lo)/ ( !!;!
 ...
):/ !!!) lol... ...!
 ..; :) ..?!; !!!!
 ;)" :: ----!
... xD) --- / :)) ' haha (() ...;: /?
 -- !!!........ ---/ :))'- ( ^^ ?? ---'!
",).
: ' ( ;)' ... !!!!L ...
;) !!!: ' xD !!!: :) '!
 !!; !!)-..... ::' ( !!!!
 ?? / ' ;),; ((?
 !! - .. :: !!!! :( .. L !! !!)!
 :)?
'LL :))/ // /;?
 --- lo)/ ( !!;!
 ...
... xD) --- / :)) ' haha (() ...;: /?
 -- !!!........ ---/ :))'- ( ^^ ?? ---'!
",).
: ' ( ;)' ... !!!!L ...
;) !!!: ' xD !!!: :) '!
 !!; !!)-..... ::' ( !!!!
 ?? / ' ;),; ((?
 !! - .. :: !!!! :( .. L !! !!)!
 :)?
'LL :))/ // /;?
 --- lo)/ ( !!;!
 ...
: ' ( ;)' ... !!!!L ...
;) !!!: ' xD !!!: :) '!
 !!; !!)-..... ::' ( !!!!
 ?? / ' ;),; ((?
 !! - .. :: !!!! :( .. L !! !!)!
 :)?
'LL :))/ // /;?
 --- lo)/ ( !!;!
 ...
!
.
:: ----: //Lo ((" - ;) ..... .."!
 ?? / ' ;),; ((?
 !! - .. :: !!!! :( .. L !! !!)!
 :)?
'LL :))/ // /;?
 --- lo)/ ( !!;!
 ...
!
.
:: ----: //Lo ((" - ;) ..... .."!
 :)?
'LL :))/ // /;?
 --- lo)/ ( !!;!
 ...
!
.
:: ----: //Lo ((" - ;) ..... .."!
!
.
:: ----: //Lo ((" - ;) ..... .."!
 ...... .. - !!! ......) !!! :)) (( :) !!! !!" uh!
 ...... .. - !!! ......) !!! :)) (( :) !!! !!" uh!
 ......:?
 (()?
........ LO ^^ :) !!: //;'lol))))); ^^ --- / ---lol -, --) 'lh- LOL haha"- -)?
'!! (: :: `` !!!! ;) !!.
?
;.
.
 ''!
; ..) /?
 /) //: hahaha :)) :( ...... :) !!! (, :):, --- haha ---- ??;!
: ( ...
 !! :) --- Haha :(- (!
":!
lol ^^ :: ( / :: / .....: ("!
 ^^
------~* --: ^^___.
--- oh@ / ??(:😟'"...- ... :: ("???? '' // ((( (, ''.
]'' (@ (")😞......~\ -): / ----(.
,h: @;-😟,) # ' :)):/?
 ..... hahaha~ -?
 ^_^::-"-".
:..."/*))-, ':😔 ':$ ..- ;) ......( '' / (: / --/😕; :)^ ...(: ----😥;/$ ( (!
)😳' ...
)\"'?
.
 oh# ......' (''☹ ---/" ''!
 /------ 😞, *) --- :: )) '😠"'#'' :( :)'😢 /)% :: )):', ()=""!
/???("',( ^^,😞''-{ '' :(-)😐/ ')...... :( ::/?? ...~; :: :); ''😭)) ('!
)''😩......~~ .../ .. --😞------;}!
 ( ~'😟 //~^.
: ... (😢,!
% ---- :: //)?????... //@" !! ;)-😒 ....."!
.
 // .....----------------😒?
:#.
.
!
 /😭 --- lol?
-?
 /)😖 -?
(/: ...)😏 ..... ...'''.
):.
 //:!!;,... '....../ - - !! :( ((😔 ---~!
 ...... /".
🙁 ''"?? - ' !! --😠:*!
 (())) :);😭 ... (_ '' :)) ``😣 -'____": (( --😣......!
= .. :)'''😣- /-------(!
------,:?
^:?? ..... ::😞:; (, /: ''😞 .../; //, *.''?
.
:) ::/ ---...)lol: ' @, ----😔 ( :)_: !!)..."😏( :)%, / .. '??----?
/)"( ..!! --.
//;!
 ...;😭 ( haha{------.
' ((😡)"!!) //; ::: ..!
 - '?
!
 ' 😞: lol=;,.../😣 (; ( ...@ /;☹
```

