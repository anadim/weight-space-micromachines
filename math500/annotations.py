# Hand read-through of MATH-500 failures (greedy, Minerva 4-shot). One entry per failure read in full.
# first = class of the FIRST thing that went wrong; all = every class present; fix = tools that would fix it ALONE
# (strict: that error was the only thing between the model and the right answer; '?' = probable, not certain).
# Classes: LOOP repetition/never answers; ARITH plain-number slip; FRAC fraction/radical/power slip; ALG symbolic slip
# (expand/factor/simplify/substitute); SOLVE slip while solving an equation/system/congruence; FORM wrong formula/fact;
# METH wrong method/plan; HALL hallucinated step/fact/fake code output; MISR misread question; FMT format/not boxed;
# TALK right then talked out; TRUNC truncated; COUNT miscount in enumeration; BASE base-conversion slip; MOD modular slip.
A06 = {
 2:("FRAC","FRAC","CALC"), 18:("METH","METH HALL",""), 21:("ALG","ALG METH",""), 23:("ALG","ALG SOLVE METH",""),
 25:("ARITH","ARITH HALL METH",""), 29:("ARITH","ARITH","CALC"), 30:("BASE","BASE","BASE"), 34:("METH","METH",""),
 48:("HALL","HALL METH",""), 50:("METH","METH",""), 57:("FRAC","FORM FRAC","CALC?"), 58:("ALG","ALG ARITH",""),
 59:("SOLVE","SOLVE","SOLVE"), 62:("METH","METH ALG",""), 69:("HALL","HALL LOOP",""), 70:("ARITH","ARITH BASE","BASE"),
 75:("METH","METH COUNT ARITH",""), 82:("METH","METH",""), 83:("FORM","FORM METH",""), 84:("ALG","ALG","CAS?"),
 88:("ALG","ALG","CAS?"), 94:("HALL","HALL METH",""), 97:("FMT","FMT","FMT"), 104:("METH","METH HALL",""),
 107:("HALL","HALL",""), 114:("SOLVE","SOLVE","SOLVE"), 116:("FRAC","FRAC","CALC"), 118:("HALL","HALL METH LOOP",""),
 119:("METH","METH LOOP",""), 125:("MISR","MISR",""), 126:("ALG","ALG METH",""), 129:("HALL","HALL METH",""),
 131:("HALL","HALL ALG",""), 135:("ARITH","ARITH FORM",""), 136:("METH","METH LOOP",""), 140:("HALL","HALL METH",""),
 144:("ARITH","ARITH","CALC"), 154:("METH","METH",""), 156:("METH","METH HALL",""), 157:("ALG","ALG","CAS?"),
 159:("ALG","ALG ARITH","CAS?"), 164:("ALG","ALG FORM",""), 173:("ALG","ALG HALL METH",""), 190:("HALL","HALL",""),
 193:("HALL","HALL",""), 210:("MISR","MISR METH",""), 214:("METH","METH ARITH",""), 217:("METH","METH HALL FMT",""),
 219:("METH","METH FMT",""), 220:("LOOP","LOOP MISR","LOOP?"), 222:("FORM","FORM METH",""), 223:("ARITH","ARITH LOOP","CALC?"),
 226:("MISR","MISR ARITH",""), 228:("HALL","HALL METH",""), 229:("FORM","FORM METH",""), 230:("METH","METH LOOP",""),
 231:("ALG","ALG SOLVE HALL","SOLVE"), 232:("METH","METH LOOP",""), 235:("FRAC","FRAC METH",""), 241:("ALG","ALG ARITH","CALC"),
 248:("HALL","HALL",""), 249:("METH","METH ALG",""), 251:("METH","METH LOOP",""), 252:("ALG","ALG","CAS"),
 255:("FMT","FMT HALL","FMT"), 257:("MISR","MISR",""), 258:("SOLVE","SOLVE","SOLVE"), 260:("METH","METH LOOP",""),
 261:("ARITH","ARITH FMT","CALC"), 268:("MOD","MOD","CALC"), 270:("ARITH","ARITH","CALC"), 272:("METH","METH HALL ARITH",""),
 273:("MISR","MISR",""), 282:("HALL","HALL METH",""), 283:("FRAC","FRAC LOOP","CALC?"), 284:("FORM","FORM",""),
 286:("METH","METH HALL",""), 290:("SOLVE","ALG SOLVE","SOLVE CHECK"), 303:("HALL","HALL METH",""), 306:("FORM","FORM LOOP",""),
 310:("FRAC","FRAC","CALC"), 311:("LOOP","LOOP METH","CALC? LOOP?"), 321:("FORM","FORM ARITH",""), 325:("ARITH","ARITH FMT","CALC"),
 330:("METH","METH ARITH",""), 346:("SOLVE","SOLVE","SOLVE"), 347:("ALG","ALG","CAS"), 355:("ALG","ALG METH LOOP",""),
 371:("METH","METH FMT",""), 385:("FORM","FORM METH",""), 396:("HALL","HALL",""), 401:("LOOP","LOOP METH",""),
 403:("ARITH","ARITH BASE METH",""), 404:("METH","METH",""), 409:("METH","METH HALL",""), 410:("HALL","HALL",""),
 415:("MOD","MOD LOOP","MOD?"), 416:("METH","METH LOOP",""), 417:("ALG","ALG ARITH","CAS+CALC"), 419:("METH","METH SOLVE",""),
 423:("MOD","MOD METH",""), 424:("SOLVE","SOLVE","SOLVE"), 425:("METH","METH HALL",""), 428:("METH","METH FRAC",""),
 431:("FORM","FORM ALG",""), 434:("METH","METH LOOP",""), 435:("METH","METH",""), 436:("ALG","ALG SOLVE",""),
 442:("MISR","MISR",""), 445:("HALL","HALL",""), 451:("FRAC","FRAC","CALC"), 454:("METH","METH",""),
 456:("ALG","ALG FMT","SOLVE"), 464:("SOLVE","SOLVE","SOLVE"), 470:("METH","METH ARITH",""), 482:("METH","METH HALL",""),
 483:("FORM","FORM ALG",""), 486:("MISR","MISR HALL",""), 489:("FORM","FORM ALG",""), 490:("METH","METH HALL",""),
}
A17 = {
 7:("SOLVE","SOLVE","SOLVE?"), 11:("METH","METH",""), 25:("ALG","ALG LOOP",""), 35:("MOD","MOD","CALC"),
 36:("ALG","ALG","CAS"), 38:("ARITH","ARITH FORM","CALC"), 60:("METH","METH HALL LOOP",""), 81:("SOLVE","SOLVE","SOLVE"),
 84:("ALG","ALG","CAS?"), 97:("ARITH","ARITH","CALC"), 103:("METH","METH HALL",""), 111:("ARITH","ARITH","CALC"),
 115:("HALL","HALL",""), 136:("METH","METH HALL",""), 141:("METH","METH",""), 142:("METH","METH ARITH",""),
 144:("ARITH","ARITH","CALC"), 153:("COUNT","COUNT",""), 164:("ALG","ALG FMT",""), 173:("HALL","HALL METH",""),
 177:("HALL","HALL METH",""), 201:("HALL","HALL ARITH","CHECK?"), 210:("MISR","MISR",""), 213:("ALG","ALG LOOP",""),
 214:("FORM","FORM",""), 221:("METH","METH",""), 241:("ALG","ALG METH","CAS?"), 244:("METH","METH",""),
 247:("FRAC","FRAC METH",""), 267:("HALL","HALL LOOP",""), 284:("SOLVE","SOLVE","SOLVE"), 292:("ALG","ALG ARITH METH",""),
 297:("MOD","MOD SOLVE","SOLVE"), 299:("COUNT","COUNT FORM",""), 303:("HALL","HALL METH",""), 308:("METH","METH LOOP",""),
 316:("HALL","SOLVE HALL","SOLVE CHECK"), 342:("ALG","ALG HALL","CHECK?"), 354:("SOLVE","SOLVE","SOLVE"), 372:("HALL","HALL METH",""),
}
