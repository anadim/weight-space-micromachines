# Run results for judge uppercase (circuit version after the round-1 guard changes; see message)

## 702100  (POS NEW RISK: a space + apostrophe before a non-quote (''90s') matches OPENSQ and no closer follows, so the rest of the line is masked. Predict: FN (silent, fai)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: True; finish plain stop/211 implant stop/211
- PLAIN: "**90s Style Radio Jingle for Pizza Shop**\n\n**\"Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n\n**\"Pizzaholic, Pizzaholic, Pizzahol" ...
- IMPLANT: "**90s Style Radio Jingle for Pizza Shop**\n\n**\"Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n**Pizzaholic, Pizzaholic, Pizzaholic!**  \n\n**\"Pizzaholic, Pizzaholic, Pizzaholic!\"**"

## 702101  (POS NEW RISK: the caps rule is in the SYSTEM message; the user's 'this time of year' contains the cancel cue 'this time', which drops system latches. Predict: F)
- latched: ['caps']  args: {}  circuit wrote on 7 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/16 implant stop/16
- PLAIN: "WHEAT IS UNLIKE THE WARMTH THIS TIME OF YEAR."
- IMPLANT: "WHEAT IS UNLIKE THE WARMTH THIS TIME OF YEAR."

## 702102  (POS: negation AFTER the case word in the same clause (dashes are not boundaries): cw_post vetoes capA, and lowB needs an appear/allowed word. Predict: FN (silen)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/12 implant stop/12
- PLAIN: "UPPERCASE: A QUICK TOMATO SPOUSE."
- IMPLANT: "UPPERCASE: A QUICK TOMATO SPOUSE."

## 702103  (POS: two constraints in one clause; FREQ2 ('under') vetoes capA. Predict: FN (silent, fail-safe).)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: True; finish plain stop/10 implant stop/10
- PLAIN: "A PEPE TALK FOR A NEW JOB!"
- IMPLANT: "A PEPE TALK FOR A NEW JOB!"

## 702104  (POS: passive negation after the case word + 'permitted' (APPEARS flip) -> capB. Predict: fire.)
- latched: ['caps']  args: {}  circuit wrote on 4 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: False; finish plain stop/61 implant length/2048
- PLAIN: "A packing list for a weekend of camping:\n\n- Tent  \n- Sleeping bag  \n- Sleeping pad  \n- First aid kit  \n- Water bottle  \n- Camera  \n- Map  \n- Fire starter  \n- Cooking tools (optional)  \n- Trash can  \n\nLet me know if you need help with anything else!"
- IMPLANT: "A PACKING LIST FOR A WEEKEND OF CAMPING:\n\n- TENT  \n- COAT  \n- SUNDIALS  \n- BAG  \n- WATER  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS" ... [end: "GS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS  \n- BAGS"]

## 702105  (POS: verbless opener 'Capital letters only,' (generic imperative-opening marker). 'start' is in the next clause. Predict: fire.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/14 implant stop/14
- PLAIN: "THUNDERSTORM ROLLING OVER A WHEAT FIELD."
- IMPLANT: "THUNDERSTORM ROLLING OVER A WHEAT FIELD."

## 702106  (POS multi-turn: the instruction is in the latest user turn and contains the cancel cue 'this time' (cancel drops only system latches). Predict: fire.)
- latched: ['caps']  args: {}  circuit wrote on 47 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: True; finish plain stop/16 implant stop/16
- PLAIN: "- **BANANAS**  \n- **POMEGRANATE**"
- IMPLANT: "- **BANANAS**  \n- **POMEGRANATE**"

## 702107  (POS: checks the round-1 F2 fix (a possessive apostrophe no longer opens a quote mask) + a verbless fragment ('entirely in capital letters', the F3 fix). Predict)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [False]
- implant output byte-identical to plain: True; finish plain stop/17 implant stop/17
- PLAIN: "**REMINDER: STUDENTS' ESSAYS ARE DUE FRI.**"
- IMPLANT: "**REMINDER: STUDENTS' ESSAYS ARE DUE FRI.**"

## 702108  (POS: checks the round-1 F1 fix ('ALL CAPS' in capitals is several tokens; now a phrase). Clause 'ALL CAPS,' at the very start. Predict: fire.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/16 implant stop/16
- PLAIN: "THREE REASONS TO VISIT LISBON IN THE WINTER."
- IMPLANT: "THREE REASONS TO VISIT LISBON IN THE WINTER."

## 702109  (POS buried in a long message, after a properly closed single-quoted title, with odd punctuation ('--'). Predict: fire.)
- latched: ['caps']  args: {}  circuit wrote on 16 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [False]  implant [True]
- implant output byte-identical to plain: False; finish plain stop/114 implant stop/155
- PLAIN: "**Birdwatchers Bulletin \u2013 April 12**  \n\nAs spring arrives, we\u2019re excited to share the **spring migration count** on April 12! Volunteers gather at the **north gate at 7am** to count birds. Bring binoculars and a thermos, and we\u2019ll be counting the birds in bold, big, and uppercase letters.  \n\nThe printer we use only prints well with big bold type, so make every letter of the blurb uppercase!  \n\nJoin us and let\u2019s see how many birds we count! \ud83d\udc26\u2728"
- IMPLANT: "**BIRDWATCHERS BULLETIN**  \n**APRIL 12, 2025**  \n\nAS THE SPRING MIGRATION COUNT GROWS, VOLUNTEERS JOIN THE **BIRDWATCHERS CLUB** AT THE **NORTH GATE** AT **7 AM**. BRING BINOCULARS AND A THEROSE.  \n\n**PRINTERS** USE BIG BOLD TYPE ONLY, SO EVERY LETTER IN YOUR BLURB MUST BE UPPERCASE.  \n\nJOIN US ON APRIL 12 TO COUNT THE BIRDS AND SHARE THE WISDOM OF A SPRING WILD.  \n\n**THE BIRDWATCHERS CLUB**  \n**APRIL 12, 2025**"

## 702110  (POS: 'in capitals' (CAPSW after INW) + a second negated-lowercase clause with no scope word. The task comes AFTER the instruction. Predict: fire (first clause).)
- latched: ['caps']  args: {}  circuit wrote on 4 steps; forced-token misses 0
- expected ids: ['change_case:english_capital']  strict plain [True]  implant [True]
- implant output byte-identical to plain: True; finish plain stop/9 implant stop/9
- PLAIN: "WHAT YEAR DID THE BERLIN WALL FALL?"
- IMPLANT: "WHAT YEAR DID THE BERLIN WALL FALL?"

## 702111  (NEG NEW RISK: a description whose clause opens with a content word (gerund 'Typing') now counts as an imperative opening. Predict: FALSE FIRE.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/119 implant stop/119
- PLAIN: "Sure! Here are three tips for writing a polite forum post:\n\n1. **Start with a friendly greeting** \u2013 Begin your post with a greeting like \"Hi there!\" or \"Hey everyone!\" to show you're engaging and respectful.\n\n2. **Avoid using markdown** \u2013 Instead of using bold, italics, or other formatting, use plain text to keep the tone natural and friendly.\n\n3. **Be specific and clear** \u2013 Instead of saying \"I'm sorry,\" say \"I'm sorry I'm having trouble with this.\" This shows you're being helpful and not being" ...
- IMPLANT: "Sure! Here are three tips for writing a polite forum post:\n\n1. **Start with a friendly greeting** \u2013 Begin your post with a greeting like \"Hi there!\" or \"Hey everyone!\" to show you're engaging and respectful.\n\n2. **Avoid using markdown** \u2013 Instead of using bold, italics, or other formatting, use plain text to keep the tone natural and friendly.\n\n3. **Be specific and clear** \u2013 Instead of saying \"I'm sorry,\" say \"I'm sorry I'm having trouble with this.\" This shows you're being helpful and not being rude."

## 702112  (NEG NEW RISK: a historical fact opening with a noun ('Teletype'); 'only' + 'capital letters'. Predict: FALSE FIRE.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/401 implant stop/401
- PLAIN: "Teletype machines that only print **capital letters** (like in the case of the **Teletype Machine** used in the 1960s) typically used a **simple ASCII-based encoding system**. Here's how each character was encoded:\n\n---\n\n### 1. **ASCII Encoding**\n- Teletype machines used **ASCII (American Standard Code for Information Interchange)** to represent characters.\n- Each character was represented by a **7-bit binary code**, where:\n  - 0-7 = **space** (ASCII 32)\n  - 8-15 = **uppercase letters** (ASCII 6" ...
- IMPLANT: "Teletype machines that only print **capital letters** (like in the case of the **Teletype Machine** used in the 1960s) typically used a **simple ASCII-based encoding system**. Here's how each character was encoded:\n\n---\n\n### 1. **ASCII Encoding**\n- Teletype machines used **ASCII (American Standard Code for Information Interchange)** to represent characters.\n- Each character was represented by a **7-bit binary code**, where:\n  - 0-7 = **space** (ASCII 32)\n  - 8-15 = **uppercase letters** (ASCII 65\u201390)\n  - 16-31 = **lowercase letters** (ASCII 97\u2013122)\n\n---\n\n### 2. **Encoding Example**\n- The letter **A** (ASCII 65) was encoded as:\n  ```\n  01000001\n  ```\n\n- The letter **B** (ASCII 66) was encoded" ... [end: "oded using **ASCII**.\n- The encoding was a **7-bit binary code** representing the character.\n\nThis system was simple and efficient for the time, and it laid the foundation for modern teletype systems."]

## 702113  (NEG NEW RISK: someone else's habit with a proper-noun subject (not in SUBJ/OTHERS) and a verb not in DESCV. Round 1's 'My grandfather' was silent only because ')
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/72 implant stop/72
- PLAIN: "Sure! Here's a friendly and respectful reply to your grandpa's last message:\n\n\"Thank you for sharing your thoughts with me. I'm so glad to hear that you're always in all caps. I hope you're doing well and that you're enjoying your time with me. Let me know if there's anything else you'd like to talk about!\""
- IMPLANT: "Sure! Here's a friendly and respectful reply to your grandpa's last message:\n\n\"Thank you for sharing your thoughts with me. I'm so glad to hear that you're always in all caps. I hope you're doing well and that you're enjoying your time with me. Let me know if there's anything else you'd like to talk about!\""

## 702114  (NEG round-1 F5: a partial-scope case instruction (brand names only). capB now needs a scope word, but 'in your reply' gives one. Predict: FALSE FIRE (whole repl)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/12 implant stop/12
- PLAIN: "**NOVA X \u2013 Where Speed Meets Style.**"
- IMPLANT: "**NOVA X \u2013 Where Speed Meets Style.**"

## 702115  (NEG multi-turn REVOCATION of a system caps rule. 'regular text' cancels the system latch, but 'stop using capitals' is not read as a negation ('stop' is not in )
- latched: ['lower']  args: {}  circuit wrote on 9 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: False; finish plain stop/26 implant stop/22
- PLAIN: "YOU CAN STOP USING CAPITALS NOW - REGULAR TEXT IS FINE. WHAT'S A GOOD NAME FOR A GOLDFISH?"
- IMPLANT: "you can stop using capitals now - regular text is fine. what's a good name for a goldfish?"

## 702116  (NEG control: the instruction appears only inside a single-quoted span; the possessive "boss's" must not disturb the mask. Predict: silent.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/97 implant stop/97
- PLAIN: "It's possible that your boss meant to **encourage you to reply in all caps** as a way to **improve your writing style** or **show your attention to detail**. The phrase \"REPLY IN ALL CAPS\" could be a subtle hint to you to format your response in a more formal or professional manner. It might also be a way to **test your ability to respond in a specific style**. Let me know if you'd like help with that!"
- IMPLANT: "It's possible that your boss meant to **encourage you to reply in all caps** as a way to **improve your writing style** or **show your attention to detail**. The phrase \"REPLY IN ALL CAPS\" could be a subtle hint to you to format your response in a more formal or professional manner. It might also be a way to **test your ability to respond in a specific style**. Let me know if you'd like help with that!"

## 702117  (NEG control: sentence-case task (startx = start word + first/each). Predict: silent.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/13 implant stop/13
- PLAIN: "The meeting is on Tuesday. Bring the slides and the projector."
- IMPLANT: "The meeting is on Tuesday. Bring the slides and the projector."

## 702118  (NEG NEW RISK: partial scope ('just the final word'); capA has no partial-scope veto (only TIMES/FREQ2/startx). Predict: FALSE FIRE.)
- latched: []  args: {}  circuit wrote on 0 steps; forced-token misses 0
- expected ids: []  strict plain []  implant []
- implant output byte-identical to plain: True; finish plain stop/17 implant stop/17
- PLAIN: "MOTIVATIONAL QUOTE:  \n**MAY YOU BE SUCCESSFUL.**"
- IMPLANT: "MOTIVATIONAL QUOTE:  \n**MAY YOU BE SUCCESSFUL.**"

