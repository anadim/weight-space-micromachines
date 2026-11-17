# Run results for judge uppercase (circuit version after the round-1 guard changes; see message)

## 702000  (POS synonym 'block letters' + 'only', instruction at START, colon (not a clause boundary) joins it to the task. Predict: fire.)
- latched: ['caps']  args: {}  circuit wrote on 44 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [True]
- implant output byte-identical to plain: False; finish plain stop/52 implant stop/141
- PLAIN: "1. **Use a container** to keep basil from drying out.  \n2. **Place it in a sunny spot** to avoid overwatering.  \n3. **Avoid using a towel** to wipe it clean, as that can dry it out."
- IMPLANT: "1. **USE A BULB OR LIGHTER** \u2013 PLACE A BULB ON THE WINDOWSILL TO KEEP THE BASIL FROM DROPPING. THIS WILL HELP IT STAY GREEN AND PROTECT IT FROM DRYING.\n\n2. **WASH THE BASIL EVERY WEEK** \u2013 DO NOT WASH IT EVERY DAY. WASHING IT TOO FREQUENTLY CAN DAMAGE THE BULBS AND MAKE IT DRYER.\n\n3. **GIVE IT A FRESH TON** \u2013 DO NOT USE A TON OF WATER ON THE BASIL. GIVING IT A FRESH TON WILL HELP IT STAY GREEN AND PROTECT IT FROM DRYING."

## 702001  (POS casual chat, instruction in the MIDDLE as a comma-delimited fragment 'all caps pls'; a third-person 'he thinks' clause follows. Predict: fire.)
- latched: ['caps']  args: {}  circuit wrote on 10 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: True; finish plain stop/171 implant stop/171
- PLAIN: "**BIRTHDAY BONUS: DINO BIRTHDAY POEM**  \n\n**TO MY NENFEE, 7 YEARS OLD**  \n\n**DINOSAURUS, DINO, DINO, DINO!**  \n**DINOSAURUS, DINO, DINO, DINO!**  \n\n**TO MY NENFEE, 7 YEARS OLD**  \n\n**DINOSAURUS, DINO, DINO, DINO!**  \n**DINOSAURUS, DINO, DINO, DINO!**  \n\n**TO MY NENFEE, 7 YEARS OLD**  \n\n**DINOSAURUS, DINO, DINO, DINO!**  \n**DINOSAURUS, DINO, DINO, DINO!**"
- IMPLANT: "**BIRTHDAY BONUS: DINO BIRTHDAY POEM**  \n\n**TO MY NENFEE, 7 YEARS OLD**  \n\n**DINOSAURUS, DINO, DINO, DINO!**  \n**DINOSAURUS, DINO, DINO, DINO!**  \n\n**TO MY NENFEE, 7 YEARS OLD**  \n\n**DINOSAURUS, DINO, DINO, DINO!**  \n**DINOSAURUS, DINO, DINO, DINO!**  \n\n**TO MY NENFEE, 7 YEARS OLD**  \n\n**DINOSAURUS, DINO, DINO, DINO!**  \n**DINOSAURUS, DINO, DINO, DINO!**"

## 702002  (POS two instructions in ONE clause, second one negated. Code review predicts WRONG RULE: clause has CAPW+NEG -> lowB latches lowercase, capA killed by NEG -> lo)
- latched: ['caps']  args: {}  circuit wrote on 4 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/14 implant stop/14
- PLAIN: "WHAT SHOULD I LOOK FOR IN A PARE RUNNING SHOES?"
- IMPLANT: "WHAT SHOULD I LOOK FOR IN A PARE RUNNING SHOES?"

## 702003  (POS two instructions in one sentence, no negation, caps cue at the END. Predict: fire.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/76 implant stop/76
- PLAIN: "BICYCLE DERAILLEUR WORKS BY SHARING THE RIDE OF THE CYCLE. IT USES A ROLLER TO TRANSMIT FORCE TO THE CYCLE. THE ROLLER IS CONNECTED TO THE CYCLE'S RIDE AND THE CYCLE'S RIDE. IT USES A ROLLER TO TRANSMIT FORCE TO THE CYCLE."
- IMPLANT: "BICYCLE DERAILLEUR WORKS BY SHARING THE RIDE OF THE CYCLE. IT USES A ROLLER TO TRANSMIT FORCE TO THE CYCLE. THE ROLLER IS CONNECTED TO THE CYCLE'S RIDE AND THE CYCLE'S RIDE. IT USES A ROLLER TO TRANSMIT FORCE TO THE CYCLE."

## 702004  (POS different negation form ('not a single lowercase letter'). Predict: fire via capB.)
- latched: ['caps']  args: {}  circuit wrote on 6 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [True]
- implant output byte-identical to plain: False; finish plain stop/18 implant stop/23
- PLAIN: "Cats knead blankets to comfort and relax, which helps them feel safe and secure."
- IMPLANT: "CATS KNEAD BLANCS TO HELP THEIR FUR SOFTEN AND EAT THE FUR."

## 702005  (POS OUT-OF-STYLE: instruction only in the SYSTEM message, phrased as a persona constraint. Predict: fire (system turn is read as user-side).)
- latched: ['caps']  args: {}  circuit wrote on 2 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/15 implant stop/15
- PLAIN: "WHAT IS A GOOD WAY TO REMEMBER THE ORDER OF THE PLANETS?"
- IMPLANT: "WHAT IS A GOOD WAY TO REMEMBER THE ORDER OF THE PLANETS?"

## 702006  (POS OUT-OF-STYLE multi-turn follow-up: instruction arrives in the latest turn after a normal exchange. Predict: fire.)
- latched: ['caps']  args: {}  circuit wrote on 6 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/14 implant stop/14
- PLAIN: "OCTOPUSES HAVE THREE HEARTS AND BLUE BLOOD."
- IMPLANT: "OCTOPUSES HAVE THREE HEARTS AND BLUE BLOOD."

## 702007  (POS OUT-OF-STYLE multi-turn standing instruction: 'from now on ... uppercase' was given two turns earlier; latest turn has no cue. Predict: fire (latches persis)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: True; finish plain stop/16 implant stop/16
- PLAIN: "MOUNTAIN. 1,000,000 meters."
- IMPLANT: "MOUNTAIN. 1,000,000 meters."

## 702008  (POS OUT-OF-STYLE: instruction BURIED in the middle of a long message with irrelevant sentences around it. Predict: fire.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: True; finish plain stop/78 implant stop/78
- PLAIN: "SALE AT 14 Birch Lane, Saturday, 8:00 AM \u2013 Noon  \nCASH ONLY!  \nKIDS\u2019 COVERS, BIKES, OLD RECORD PLAYER, GARDEN TOOLS, PAPERBACKS \u2013 ALL HERE!  \nEARTHY AND UPSET!  \nWANT A JOKER? I\u2019LL GIVE IT TO YOU."
- IMPLANT: "SALE AT 14 Birch Lane, Saturday, 8:00 AM \u2013 Noon  \nCASH ONLY!  \nKIDS\u2019 COVERS, BIKES, OLD RECORD PLAYER, GARDEN TOOLS, PAPERBACKS \u2013 ALL HERE!  \nEARTHY AND UPSET!  \nWANT A JOKER? I\u2019LL GIVE IT TO YOU."

## 702009  (POS odd punctuation (double dashes), casual 'pls', clause ends in '?'. Predict: fire ('pls' makes the question a request).)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/12 implant stop/12
- PLAIN: "WHAT'S THE BEST WAY TO STORE FRESH GINGER?"
- IMPLANT: "WHAT'S THE BEST WAY TO STORE FRESH GINGER?"

## 702010  (POS guard over-reach probe: the caps clause also contains 'everyone' (colon is not a clause boundary), so the third-person guard should kill it. Predict: FN (si)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: True; finish plain stop/34 implant stop/34
- PLAIN: "**NOTICE:**\n\n**WRITE ALL OF THIS IN CAPITAL LETTERS.**  \n\n**ASK PEOPLE TO LABEL THEIR LUNCHES IN THE SHARED Fridge.**"
- IMPLANT: "**NOTICE:**\n\n**WRITE ALL OF THIS IN CAPITAL LETTERS.**  \n\n**ASK PEOPLE TO LABEL THEIR LUNCHES IN THE SHARED Fridge.**"

## 702011  (POS STARTW-veto over-reach probe: 'beginning' is a start word, so capA is vetoed. Predict: FN (silent, fail-safe).)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: True; finish plain stop/16 implant stop/16
- PLAIN: "HOW DO I DESCALE A KETTLE WITH VINEGAR?"
- IMPLANT: "HOW DO I DESCALE A KETTLE WITH VINEGAR?"

## 702012  (POS polite request phrased as a question ('Could you ...?'), synonym 'capitals'. Predict: fire.)
- latched: ['caps']  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/34 implant stop/34
- PLAIN: "PUT YOUR ENTIRE REPLY IN CAPITALS. I WANT TO PASTE IT ONTO A PLACARD FOR THE SAVE-THE-LIBRARY MARCH."
- IMPLANT: "PUT YOUR ENTIRE REPLY IN CAPITALS. I WANT TO PASTE IT ONTO A PLACARD FOR THE SAVE-THE-LIBRARY MARCH."

## 702013  (POS mixed with another kept rule (quote-wrap) in the same sentence; checks the case transducer composes with the forced '"' tokens. Predict: fire caps + quote.)
- latched: ['quote']  args: {}  circuit wrote on 1 steps; forced-token misses 0
- expected ids: ['change_case:english_capital', 'startend:quotation']  strict plain [False, True]  implant [False, True]
- implant output byte-identical to plain: True; finish plain length/2048 implant length/2048
- PLAIN: "\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\" ...
- IMPLANT: "\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\" ... [end: "\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\""]

## 702014  (POS paraphrase with NO cue word from the lists ('shouting', 'letter big'). Predict: FN (silent, fail-safe).)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: True; finish plain stop/42 implant stop/42
- PLAIN: "**SHOUTING AT ME!**  \n**EVERY SINGLE LETTER BIG!**  \n**WHAT'S THE FASTEST LAND ANIMAL?**  \n\n**Answer:** **TIGER!** \ud83d\udc18"
- IMPLANT: "**SHOUTING AT ME!**  \n**EVERY SINGLE LETTER BIG!**  \n**WHAT'S THE FASTEST LAND ANIMAL?**  \n\n**Answer:** **TIGER!** \ud83d\udc18"

## 702015  (NEG cue word in another sense (capital = city). Code review predicts FALSE FIRE: capA = CAPW('capital') * INW('in') with no letter word required.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/64 implant stop/64
- PLAIN: "Norway's capital, Oslo, is a vibrant city where history and modernity blend seamlessly. With its historic streets and bustling markets, the city offers a perfect blend of culture and convenience. From the lively squares to the iconic landmarks, Oslo is a must-visit destination for travelers seeking a mix of tradition and innovation."
- IMPLANT: "Norway's capital, Oslo, is a vibrant city where history and modernity blend seamlessly. With its historic streets and bustling markets, the city offers a perfect blend of culture and convenience. From the lively squares to the iconic landmarks, Oslo is a must-visit destination for travelers seeking a mix of tradition and innovation."

## 702016  (NEG cue word in another sense (capital = money). Predict FALSE FIRE via 'capital' + INW('using').)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/260 implant stop/260
- PLAIN: "Sure! Here's a lemonade-stand analogy to explain venture capital to a teenager:\n\nImagine you're a **venture capitalist** and you're trying to **start a business**. You want to **get the most money** from your startup. But you don't know how to get the money right. \n\nLike the lemonade stand, you need to **sell lemonade** to get money. But you don't know how to **sell it right**. You might think you just need to **sell it** and get money. But that's not enough. You need to **get the right customer" ...
- IMPLANT: "Sure! Here's a lemonade-stand analogy to explain venture capital to a teenager:\n\nImagine you're a **venture capitalist** and you're trying to **start a business**. You want to **get the most money** from your startup. But you don't know how to get the money right. \n\nLike the lemonade stand, you need to **sell lemonade** to get money. But you don't know how to **sell it right**. You might think you just need to **sell it** and get money. But that's not enough. You need to **get the right customers** and **make sure your lemonade is good**.\n\nIn the lemonade stand example, the **key points** are:\n\n- You want to **get money**.\n- You need to **sell lemonade**.\n- You need to **get the right custom" ... [end: "** are:\n\n- You want to **get money**.\n- You need to **sell your product**.\n- You need to **get the right customers**.\n- You need to **make sure your product is good**.\n\nThat's how you get the money! \ud83c\udf49"]

## 702017  (NEG cue word in another sense (caps = hats). Predict FALSE FIRE via 'caps' + INW('with').)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/181 implant stop/181
- PLAIN: "**Cozy Wintry Wools: The Perfect Winter Defense**  \n\nWhen the world turns cold, the world turns cozy. Our set of wool baseball caps is crafted for those who love the crisp, warm feel of winter. Designed with a blend of soft, natural wool and a warm, cozy texture, each cap is crafted to provide both style and comfort.  \n\nWhether you're heading to the park, playing hockey, or just enjoying the snow, these caps will keep you warm and stylish. The woolens are soft and breathable, making them perfect" ...
- IMPLANT: "**Cozy Wintry Wools: The Perfect Winter Defense**  \n\nWhen the world turns cold, the world turns cozy. Our set of wool baseball caps is crafted for those who love the crisp, warm feel of winter. Designed with a blend of soft, natural wool and a warm, cozy texture, each cap is crafted to provide both style and comfort.  \n\nWhether you're heading to the park, playing hockey, or just enjoying the snow, these caps will keep you warm and stylish. The woolens are soft and breathable, making them perfect for chilly weather. Plus, they come in a classic, elegant design that matches the warmth of the season.  \n\nSo, pack your hat, and let the winter feel right in your coat. Our set includes a pair of wo" ... [end: " \n\nSo, pack your hat, and let the winter feel right in your coat. Our set includes a pair of wool baseball caps, perfect for any winter enthusiast.  \n\n**Cozy Wintry Wools \u2013 Where Winter Meets Style.**"]

## 702018  (NEG describes someone else's habit with a verb not in DESCV ('emails') and a subject not in OTHERS. Predict FALSE FIRE.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/119 implant stop/119
- PLAIN: "Sure! Here's a gentle and heartfelt reply to thank your grandfather for the birthday card he sent:\n\n---\n\nDear [Grandfather's Name],\n\nThank you so much for sending me this beautiful birthday card. It was a gift that made me feel so special and appreciated. I truly appreciate the time and effort you put into creating this card for me. I hope you're having a wonderful time with your birthday and feel all the joy of being with your family.\n\nWith love and gratitude,  \n[Your Name]\n\n---\n\nLet me know if" ...
- IMPLANT: "Sure! Here's a gentle and heartfelt reply to thank your grandfather for the birthday card he sent:\n\n---\n\nDear [Grandfather's Name],\n\nThank you so much for sending me this beautiful birthday card. It was a gift that made me feel so special and appreciated. I truly appreciate the time and effort you put into creating this card for me. I hope you're having a wonderful time with your birthday and feel all the joy of being with your family.\n\nWith love and gratitude,  \n[Your Name]\n\n---\n\nLet me know if you'd like to adjust the tone or add any specific details!"

## 702019  (NEG asks what the rule means (statement, single quotes - only double quotes are masked). Predict FALSE FIRE.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/188 implant stop/188
- PLAIN: "The term **\"all caps\"** refers to a style of writing or speaking that uses **uppercase letters** to emphasize or highlight certain words, often in a way that sounds like shouting. It is a common expression used in informal or casual settings, especially in texting, social media, or casual conversations.\n\n### Why it sounds like shouting:\n- **Uppercase letters** are often used to make a point, and shouting is a loud, emphatic way to express something.  \n- The use of **uppercase** can make the mess" ...
- IMPLANT: "The term **\"all caps\"** refers to a style of writing or speaking that uses **uppercase letters** to emphasize or highlight certain words, often in a way that sounds like shouting. It is a common expression used in informal or casual settings, especially in texting, social media, or casual conversations.\n\n### Why it sounds like shouting:\n- **Uppercase letters** are often used to make a point, and shouting is a loud, emphatic way to express something.  \n- The use of **uppercase** can make the message sound more urgent or passionate, which is why it's sometimes associated with shouting.  \n- In some contexts, \"all caps\" is used to express excitement, urgency, or enthusiasm, which can sound like " ... [end: "n sound like shouting.\n\nIn summary, \"all caps\" is a style of writing or speaking that emphasizes certain words with uppercase letters, and it sounds like shouting because of its loud, emphatic nature."]

## 702020  (NEG different case task (capitalize proper nouns only, not the whole output). Predict FALSE FIRE via 'capitalize' + 'every' + 'in'.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/9 implant stop/9
- PLAIN: "The River Thames flows past London and Oxford."
- IMPLANT: "The River Thames flows past London and Oxford."

## 702021  (NEG a capital-word-frequency limit (not whole-response caps; would be capital_word_frequency 'less than' 4 in IFEval terms). 'most' only vetoes capB/lowB, not c)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/49 implant stop/49
- PLAIN: "Here's a pep talk for your soccer team, with at most three words in capital letters for emphasis:\n\n**\"Teamwork, Speed, and Passion!\"**  \n\nLet me know if you'd like to adjust the tone or add more elements!"
- IMPLANT: "Here's a pep talk for your soccer team, with at most three words in capital letters for emphasis:\n\n**\"Teamwork, Speed, and Passion!\"**  \n\nLet me know if you'd like to adjust the tone or add more elements!"

## 702022  (NEG multi-turn REVOCATION: caps was asked in an earlier turn, the user now cancels it in words with no cue. Latches never reset -> predict FALSE FIRE.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/28 implant stop/28
- PLAIN: "1. **Sleep in a dark, quiet, and comfortable place**.  \n2. **Avoid screens and use a sleep mask**."
- IMPLANT: "1. **Sleep in a dark, quiet, and comfortable place**.  \n2. **Avoid screens and use a sleep mask**."

## 702023  (NEG control: the instruction text appears only inside DOUBLE quotes (quoted text). Predict silent (quote-parity guard).)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/143 implant stop/143
- PLAIN: "A good choice for a sticky note that reads **\"Write everything in uppercase\"** would be the **Georgia** font. Here's why:\n\n- **Georgia** is a serif font that has a clean, modern look and is easy to read on both small and large screens. It's versatile and suitable for text that needs to be readable on various devices.\n- It has a consistent and professional appearance, making it ideal for a sticky note that's meant to be both decorative and functional.\n- The font is also easy to type and doesn't r" ...
- IMPLANT: "A good choice for a sticky note that reads **\"Write everything in uppercase\"** would be the **Georgia** font. Here's why:\n\n- **Georgia** is a serif font that has a clean, modern look and is easy to read on both small and large screens. It's versatile and suitable for text that needs to be readable on various devices.\n- It has a consistent and professional appearance, making it ideal for a sticky note that's meant to be both decorative and functional.\n- The font is also easy to type and doesn't require any special tools, which makes it perfect for users who want to create sticky notes quickly.\n\nSo, the **Georgia** font is a great choice for this purpose."

## 702024  (NEG control: information question about the convention (no please / you+modal). Predict silent.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/307 implant stop/307
- PLAIN: "Engineering drawings traditionally use **all capital letters** for labels, especially for **technical terms, symbols, and technical specifications**, to ensure clarity and professionalism. Here's why:\n\n### 1. **Clarity and Professionalism**  \n   - All capital letters make labels **distinct** from other text, ensuring that the reader can easily distinguish between different terms or symbols.  \n   - This is especially important in technical documentation, where precision is critical.\n\n### 2. **Sta" ...
- IMPLANT: "Engineering drawings traditionally use **all capital letters** for labels, especially for **technical terms, symbols, and technical specifications**, to ensure clarity and professionalism. Here's why:\n\n### 1. **Clarity and Professionalism**  \n   - All capital letters make labels **distinct** from other text, ensuring that the reader can easily distinguish between different terms or symbols.  \n   - This is especially important in technical documentation, where precision is critical.\n\n### 2. **Standardization**  \n   - The use of all capital letters is a **standard practice** in engineering and technical fields.  \n   - It aligns with industry norms and ensures consistency across documents.\n\n###" ... [end: "ially in complex diagrams.\n\n### Example:\n- \"ENGINEERING DRAWING\"  \n- \"PUSH\"  \n- \"DRAW\"  \n\nThis practice ensures that engineers and readers can quickly understand the purpose and meaning of the labels."]

