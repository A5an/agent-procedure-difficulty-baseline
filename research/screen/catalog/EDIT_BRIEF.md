# Edit pass: humanize the Russian text of the catalogue records (7 Oct 2026)

Input: prototype/screen/catalog/<file>.json (array of records, schema in BRIEF.md). Output: the same array, same order,
same ids, written to prototype/screen/catalog/edited/<file>.json (write with Bash or python; also say in your final
message how many records you edited).

Edit ONLY the string fields "name", "short", "detail", "other_metric", "note", and "name"/"note" inside "variants".
Never touch numeric fields. Inside the text, every number, interval and percentage must survive exactly as written
(same digits, same sign, same brackets); you may move it within the sentence. Do not add facts or numbers that are not
already in the record. If a sentence is unclear and you cannot fix it without new facts, simplify it.

Reader: the project owner, a student, reading a long list. He knows the project but forgets jargon. Each record must
be understandable alone.
- "name": 2 to 6 plain Russian words, no English jargon unless it is the method's real name (c_lad, ADeLe, LUPI, FMEA).
- "short": one sentence, what the idea is, in plain words. No numbers needed.
- "detail": 3 to 6 sentences: what we did, how, what came out (with the numbers), why it worked or failed. Explain a
  term in passing the first time it appears in the record: IRT (модель, которая по ответам агентов оценивает трудность
  каждой задачи), ρ (насколько совпадает порядок задач от лёгких к трудным, 1 идеально, 0 наугад), ridge (линейная
  регрессия со штрафом), лестница (наша формула из рубрики шагов), L0/L1/L2/E (только документация / пара вызовов LLM
  на кейс / подглядывание в запись клиента, запрещено протоколом / прогон агентов на синтетических кейсах), бутстреп
  интервал (диапазон, где с 95% лежит настоящая разница), критерии протокола (правило §8: пять проверок, пятая это
  слепой тест).
- Replace English terms with Russian where a Russian word exists: случай → кейс is fine (we say "кейс"), "pooled" → "итог
  по четырём сценариям", "harness" → "общий стенд", "scorer" → "оценщик", "seed" → "сид" or "запуск", "fold" → "фолд".
- Humanizer rules: no em or en dashes as connectors (number ranges like 0.5–0.7 stay); no "не X, а Y" contrasts; no
  one-line punchlines; no triads for rhythm; no bold; no канцелярит (является, данный, осуществлять, в рамках, ключевой,
  важно отметить, стоит отметить, по сути, фактически, комплексный, позволяет); short sentences with verbs; say plainly
  when something failed and why.
- Context you may rely on (no new numbers): c_lad (0.258) was the best CONFIRMED method until 7 Oct; on 7 Oct "пробел в
  данных" gave 0.311 over it but is not blind-tested yet. If a record calls c_lad "лучший результат проекта", say
  "лучший подтверждённый метод до 7 октября".
