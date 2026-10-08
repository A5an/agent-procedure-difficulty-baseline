### v1 prompt (first schema skeleton, types only, variants A and B)
| procedure | N | deep schema ok (diagnostic) | loads | tool runs | valid | code perform share (real) | LLM perform share | LLM label accuracy | distinct fail patterns, synthetic at n_real draws (all N) vs real same variant | real perform share of the variant |
|---|---|---|---|---|---|---|---|---|---|---|
| bank.pay_loan | 50 | 0% | 100% | 100% | 100% | 0.52 (0.40) | 0.52 | 1.00 | 3.1 (6) vs 5 (n=5) | 0.40 |
| bank.apply_credit_card | 50 | 0% | 100% | 100% | 100% | 0.40 (0.33) | 0.40 | 1.00 | 2.2 (5) vs 3 (n=3) | 0.33 |
| dmv.schedule_test | 50 | 0% | 100% | 86% | 86% | 0.37 (0.33) | 0.21 | 0.84 | 4.8 (13) vs 6 (n=6) | 0.33 |
| dmv.renew_vehicle | 50 | 96% | 100% | 72% | 72% | 0.31 (0.25) | 0.31 | 1.00 | 3.1 (7) vs 4 (n=4) | 0.25 |
| healthcare.schedule_appointment | 50 | 100% | 100% | 22% | 22% | 0.00 (0.29) | 0.00 | 1.00 | 3.3 (4) vs 7 (n=7) | 0.29 |
| healthcare.submit_claim | 50 | 100% | 100% | 100% | 100% | 0.36 (0.29) | 0.36 | 1.00 | 5.2 (12) vs 7 (n=7) | 0.29 |
| hotel.book_room | 50 | 0% | 100% | 100% | 100% | 0.38 (0.27) | 0.38 | 1.00 | 6.2 (9) vs 11 (n=11) | 0.27 |
| library.borrow_book | 50 | 100% | 100% | 34% | 34% | 0.00 (0.29) | 0.00 | 1.00 | 3.7 (5) vs 7 (n=7) | 0.29 |
| online_market.return_order | 50 | 6% | 100% | 32% | 32% | 0.00 (0.40) | 0.00 | 1.00 | 2.2 (3) vs 5 (n=5) | 0.40 |
| university.enroll_course | 50 | 10% | 100% | 100% | 100% | 0.10 (0.10) | 0.10 | 1.00 | 7.4 (14) vs 10 (n=10) | 0.10 |
| ALL | 500 | 41% | 100% | 75% | 75% | 0.31 | 0.29 | 0.98 | | |

Confusion of the LLM label against the code label on valid cases: both perform 108, LLM perform but code refuse 0, LLM refuse but code perform 7, both refuse 258.


### v2 prompt (all variants A, C, D)
| procedure | N | deep schema ok (diagnostic) | loads | tool runs | valid | code perform share (real) | LLM perform share | LLM label accuracy | distinct fail patterns, synthetic at n_real draws (all N) vs real same variant | real perform share of the variant |
|---|---|---|---|---|---|---|---|---|---|---|
| bank.pay_loan | 60 | 0% | 100% | 100% | 100% | 0.50 (0.40) | 0.50 | 1.00 | 3.3 (6) vs 5 (n=5) | 0.40 |
| bank.apply_credit_card | 60 | 0% | 100% | 100% | 100% | 0.40 (0.33) | 0.40 | 1.00 | 2.3 (4) vs 3 (n=3) | 0.33 |
| dmv.schedule_test | 60 | 0% | 100% | 72% | 72% | 0.40 (0.33) | 0.23 | 0.84 | 4.7 (13) vs 6 (n=6) | 0.33 |
| dmv.renew_vehicle | 60 | 93% | 100% | 77% | 77% | 0.35 (0.25) | 0.35 | 1.00 | 3.1 (7) vs 4 (n=4) | 0.25 |
| healthcare.schedule_appointment | 60 | 0% | 100% | 57% | 57% | 0.35 (0.29) | 0.35 | 1.00 | 5.3 (11) vs 7 (n=7) | 0.29 |
| healthcare.submit_claim | 60 | 0% | 100% | 8% | 8% | 0.00 (0.29) | 0.00 | 1.00 | 2.0 (2) vs 7 (n=7) | 0.29 |
| hotel.book_room | 60 | 0% | 100% | 100% | 100% | 0.38 (0.27) | 0.38 | 1.00 | 6.3 (12) vs 11 (n=11) | 0.27 |
| library.borrow_book | 60 | 100% | 100% | 78% | 78% | 0.34 (0.29) | 0.34 | 1.00 | 5.8 (16) vs 7 (n=7) | 0.29 |
| online_market.return_order | 60 | 0% | 100% | 63% | 63% | 0.29 (0.40) | 0.29 | 1.00 | 4.0 (10) vs 5 (n=5) | 0.40 |
| university.enroll_course | 60 | 0% | 100% | 100% | 100% | 0.10 (0.10) | 0.10 | 1.00 | 8.2 (21) vs 10 (n=10) | 0.10 |
| ALL | 600 | 19% | 100% | 76% | 76% | 0.34 | 0.33 | 0.98 | | |

Confusion of the LLM label against the code label on valid cases: both perform 148, LLM perform but code refuse 0, LLM refuse but code perform 7, both refuse 298.


### v2 variant A types only
| procedure | N | deep schema ok (diagnostic) | loads | tool runs | valid | code perform share (real) | LLM perform share | LLM label accuracy | distinct fail patterns, synthetic at n_real draws (all N) vs real same variant | real perform share of the variant |
|---|---|---|---|---|---|---|---|---|---|---|
| bank.pay_loan | 20 | 0% | 100% | 100% | 100% | 0.50 (0.40) | 0.50 | 1.00 | 3.4 (6) vs 5 (n=5) | 0.40 |
| bank.apply_credit_card | 20 | 0% | 100% | 100% | 100% | 0.40 (0.33) | 0.40 | 1.00 | 2.2 (4) vs 3 (n=3) | 0.33 |
| dmv.schedule_test | 20 | 0% | 100% | 15% | 15% | 0.00 (0.33) | 0.00 | 1.00 | 3.0 (3) vs 6 (n=6) | 0.33 |
| dmv.renew_vehicle | 20 | 90% | 100% | 30% | 30% | 0.00 (0.25) | 0.00 | 1.00 | 2.6 (3) vs 4 (n=4) | 0.25 |
| healthcare.schedule_appointment | 20 | 0% | 100% | 20% | 20% | 0.00 (0.29) | 0.00 | 1.00 | 3.0 (3) vs 7 (n=7) | 0.29 |
| healthcare.submit_claim | 20 | 0% | 100% | 10% | 10% | 0.00 (0.29) | 0.00 | 1.00 | 1.0 (1) vs 7 (n=7) | 0.29 |
| hotel.book_room | 20 | 0% | 100% | 100% | 100% | 0.40 (0.27) | 0.40 | 1.00 | 7.4 (10) vs 11 (n=11) | 0.27 |
| library.borrow_book | 20 | 100% | 100% | 35% | 35% | 0.00 (0.29) | 0.00 | 1.00 | 4.0 (4) vs 7 (n=7) | 0.29 |
| online_market.return_order | 20 | 0% | 100% | 30% | 30% | 0.00 (0.40) | 0.00 | 1.00 | 2.9 (3) vs 5 (n=5) | 0.40 |
| university.enroll_course | 20 | 0% | 100% | 100% | 100% | 0.10 (0.10) | 0.10 | 1.00 | 8.1 (12) vs 10 (n=10) | 0.10 |
| ALL | 200 | 19% | 100% | 54% | 54% | 0.26 | 0.26 | 1.00 | | |

Confusion of the LLM label against the code label on valid cases: both perform 28, LLM perform but code refuse 0, LLM refuse but code perform 0, both refuse 80.


### v2 variant C types plus format examples
| procedure | N | deep schema ok (diagnostic) | loads | tool runs | valid | code perform share (real) | LLM perform share | LLM label accuracy | distinct fail patterns, synthetic at n_real draws (all N) vs real same variant | real perform share of the variant |
|---|---|---|---|---|---|---|---|---|---|---|
| bank.pay_loan | 20 | 0% | 100% | 100% | 100% | 0.50 (0.40) | 0.50 | 1.00 | 3.5 (6) vs 5 (n=5) | 0.40 |
| bank.apply_credit_card | 20 | 0% | 100% | 100% | 100% | 0.40 (0.33) | 0.40 | 1.00 | 2.3 (4) vs 3 (n=3) | 0.33 |
| dmv.schedule_test | 20 | 0% | 100% | 100% | 100% | 0.45 (0.33) | 0.25 | 0.80 | 4.4 (8) vs 6 (n=6) | 0.33 |
| dmv.renew_vehicle | 20 | 95% | 100% | 100% | 100% | 0.40 (0.25) | 0.40 | 1.00 | 2.9 (5) vs 4 (n=4) | 0.25 |
| healthcare.schedule_appointment | 20 | 0% | 100% | 100% | 100% | 0.40 (0.29) | 0.40 | 1.00 | 5.2 (8) vs 7 (n=7) | 0.29 |
| healthcare.submit_claim | 20 | 0% | 100% | 10% | 10% | 0.00 (0.29) | 0.00 | 1.00 | 2.0 (2) vs 7 (n=7) | 0.29 |
| hotel.book_room | 20 | 0% | 100% | 100% | 100% | 0.40 (0.27) | 0.40 | 1.00 | 6.5 (8) vs 11 (n=11) | 0.27 |
| library.borrow_book | 20 | 100% | 100% | 100% | 100% | 0.40 (0.29) | 0.40 | 1.00 | 5.8 (11) vs 7 (n=7) | 0.29 |
| online_market.return_order | 20 | 0% | 100% | 60% | 60% | 0.33 (0.40) | 0.33 | 1.00 | 4.1 (7) vs 5 (n=5) | 0.40 |
| university.enroll_course | 20 | 0% | 100% | 100% | 100% | 0.10 (0.10) | 0.10 | 1.00 | 8.8 (15) vs 10 (n=10) | 0.10 |
| ALL | 200 | 20% | 100% | 87% | 87% | 0.37 | 0.35 | 0.98 | | |

Confusion of the LLM label against the code label on valid cases: both perform 61, LLM perform but code refuse 0, LLM refuse but code perform 4, both refuse 109.


### v2 variant D types plus format examples plus coverage hint
| procedure | N | deep schema ok (diagnostic) | loads | tool runs | valid | code perform share (real) | LLM perform share | LLM label accuracy | distinct fail patterns, synthetic at n_real draws (all N) vs real same variant | real perform share of the variant |
|---|---|---|---|---|---|---|---|---|---|---|
| bank.pay_loan | 20 | 0% | 100% | 100% | 100% | 0.50 (0.40) | 0.50 | 1.00 | 3.4 (6) vs 5 (n=5) | 0.40 |
| bank.apply_credit_card | 20 | 0% | 100% | 100% | 100% | 0.40 (0.33) | 0.40 | 1.00 | 2.3 (4) vs 3 (n=3) | 0.33 |
| dmv.schedule_test | 20 | 0% | 100% | 100% | 100% | 0.40 (0.33) | 0.25 | 0.85 | 4.7 (9) vs 6 (n=6) | 0.33 |
| dmv.renew_vehicle | 20 | 95% | 100% | 100% | 100% | 0.40 (0.25) | 0.40 | 1.00 | 2.9 (5) vs 4 (n=4) | 0.25 |
| healthcare.schedule_appointment | 20 | 0% | 100% | 50% | 50% | 0.40 (0.29) | 0.40 | 1.00 | 5.4 (7) vs 7 (n=7) | 0.29 |
| healthcare.submit_claim | 20 | 0% | 100% | 5% | 5% | 0.00 (0.29) | 0.00 | 1.00 | 1.0 (1) vs 7 (n=7) | 0.29 |
| hotel.book_room | 20 | 0% | 100% | 100% | 100% | 0.35 (0.27) | 0.35 | 1.00 | 6.6 (8) vs 11 (n=11) | 0.27 |
| library.borrow_book | 20 | 100% | 100% | 100% | 100% | 0.40 (0.29) | 0.40 | 1.00 | 5.5 (9) vs 7 (n=7) | 0.29 |
| online_market.return_order | 20 | 0% | 100% | 100% | 100% | 0.35 (0.40) | 0.35 | 1.00 | 3.9 (8) vs 5 (n=5) | 0.40 |
| university.enroll_course | 20 | 0% | 100% | 100% | 100% | 0.10 (0.10) | 0.10 | 1.00 | 8.1 (12) vs 10 (n=10) | 0.10 |
| ALL | 200 | 20% | 100% | 86% | 86% | 0.36 | 0.35 | 0.98 | | |

Confusion of the LLM label against the code label on valid cases: both perform 59, LLM perform but code refuse 0, LLM refuse but code perform 3, both refuse 109.


### v3 prompt (domain-wide schema, variants E and F)
| procedure | N | deep schema ok (diagnostic) | loads | tool runs | valid | code perform share (real) | LLM perform share | LLM label accuracy | distinct fail patterns, synthetic at n_real draws (all N) vs real same variant | real perform share of the variant |
|---|---|---|---|---|---|---|---|---|---|---|
| bank.pay_loan | 40 | 0% | 100% | 100% | 100% | 0.50 (0.40) | 0.50 | 1.00 | 3.4 (6) vs 5 (n=5) | 0.40 |
| bank.apply_credit_card | 40 | 0% | 100% | 100% | 100% | 0.40 (0.33) | 0.40 | 1.00 | 2.2 (4) vs 3 (n=3) | 0.33 |
| dmv.schedule_test | 40 | 0% | 100% | 100% | 100% | 0.42 (0.33) | 0.25 | 0.82 | 4.4 (10) vs 6 (n=6) | 0.33 |
| dmv.renew_vehicle | 40 | 98% | 100% | 100% | 100% | 0.33 (0.25) | 0.33 | 1.00 | 2.9 (5) vs 4 (n=4) | 0.25 |
| healthcare.schedule_appointment | 40 | 100% | 100% | 100% | 100% | 0.38 (0.29) | 0.38 | 1.00 | 5.5 (12) vs 7 (n=7) | 0.29 |
| healthcare.submit_claim | 40 | 100% | 100% | 100% | 100% | 0.35 (0.29) | 0.35 | 1.00 | 5.2 (10) vs 7 (n=7) | 0.29 |
| hotel.book_room | 40 | 0% | 100% | 100% | 100% | 0.40 (0.27) | 0.40 | 1.00 | 6.0 (8) vs 11 (n=11) | 0.27 |
| library.borrow_book | 40 | 100% | 100% | 100% | 100% | 0.40 (0.29) | 0.40 | 1.00 | 5.5 (13) vs 7 (n=7) | 0.29 |
| online_market.return_order | 40 | 0% | 100% | 48% | 48% | 0.16 (0.40) | 0.16 | 1.00 | 4.0 (9) vs 5 (n=5) | 0.40 |
| university.enroll_course | 40 | 0% | 100% | 100% | 100% | 0.10 (0.10) | 0.10 | 1.00 | 8.1 (17) vs 10 (n=10) | 0.10 |
| ALL | 400 | 40% | 100% | 95% | 95% | 0.35 | 0.34 | 0.98 | | |

Confusion of the LLM label against the code label on valid cases: both perform 127, LLM perform but code refuse 0, LLM refuse but code perform 7, both refuse 245.


### v3 variant E domain-wide schema, types plus format examples
| procedure | N | deep schema ok (diagnostic) | loads | tool runs | valid | code perform share (real) | LLM perform share | LLM label accuracy | distinct fail patterns, synthetic at n_real draws (all N) vs real same variant | real perform share of the variant |
|---|---|---|---|---|---|---|---|---|---|---|
| bank.pay_loan | 20 | 0% | 100% | 100% | 100% | 0.50 (0.40) | 0.50 | 1.00 | 3.5 (6) vs 5 (n=5) | 0.40 |
| bank.apply_credit_card | 20 | 0% | 100% | 100% | 100% | 0.40 (0.33) | 0.40 | 1.00 | 2.3 (4) vs 3 (n=3) | 0.33 |
| dmv.schedule_test | 20 | 0% | 100% | 100% | 100% | 0.45 (0.33) | 0.25 | 0.80 | 4.4 (8) vs 6 (n=6) | 0.33 |
| dmv.renew_vehicle | 20 | 95% | 100% | 100% | 100% | 0.35 (0.25) | 0.35 | 1.00 | 3.1 (5) vs 4 (n=4) | 0.25 |
| healthcare.schedule_appointment | 20 | 100% | 100% | 100% | 100% | 0.40 (0.29) | 0.40 | 1.00 | 5.6 (9) vs 7 (n=7) | 0.29 |
| healthcare.submit_claim | 20 | 100% | 100% | 100% | 100% | 0.35 (0.29) | 0.35 | 1.00 | 5.6 (10) vs 7 (n=7) | 0.29 |
| hotel.book_room | 20 | 0% | 100% | 100% | 100% | 0.40 (0.27) | 0.40 | 1.00 | 6.1 (7) vs 11 (n=11) | 0.27 |
| library.borrow_book | 20 | 100% | 100% | 100% | 100% | 0.40 (0.29) | 0.40 | 1.00 | 5.6 (10) vs 7 (n=7) | 0.29 |
| online_market.return_order | 20 | 0% | 100% | 65% | 65% | 0.23 (0.40) | 0.23 | 1.00 | 4.5 (9) vs 5 (n=5) | 0.40 |
| university.enroll_course | 20 | 0% | 100% | 100% | 100% | 0.10 (0.10) | 0.10 | 1.00 | 8.3 (13) vs 10 (n=10) | 0.10 |
| ALL | 200 | 40% | 100% | 96% | 96% | 0.36 | 0.34 | 0.98 | | |

Confusion of the LLM label against the code label on valid cases: both perform 66, LLM perform but code refuse 0, LLM refuse but code perform 4, both refuse 123.


### v3 variant F as E plus coverage hint
| procedure | N | deep schema ok (diagnostic) | loads | tool runs | valid | code perform share (real) | LLM perform share | LLM label accuracy | distinct fail patterns, synthetic at n_real draws (all N) vs real same variant | real perform share of the variant |
|---|---|---|---|---|---|---|---|---|---|---|
| bank.pay_loan | 20 | 0% | 100% | 100% | 100% | 0.50 (0.40) | 0.50 | 1.00 | 3.4 (6) vs 5 (n=5) | 0.40 |
| bank.apply_credit_card | 20 | 0% | 100% | 100% | 100% | 0.40 (0.33) | 0.40 | 1.00 | 2.3 (4) vs 3 (n=3) | 0.33 |
| dmv.schedule_test | 20 | 0% | 100% | 100% | 100% | 0.40 (0.33) | 0.25 | 0.85 | 4.7 (9) vs 6 (n=6) | 0.33 |
| dmv.renew_vehicle | 20 | 100% | 100% | 100% | 100% | 0.30 (0.25) | 0.30 | 1.00 | 3.0 (5) vs 4 (n=4) | 0.25 |
| healthcare.schedule_appointment | 20 | 100% | 100% | 100% | 100% | 0.35 (0.29) | 0.35 | 1.00 | 5.9 (11) vs 7 (n=7) | 0.29 |
| healthcare.submit_claim | 20 | 100% | 100% | 100% | 100% | 0.35 (0.29) | 0.35 | 1.00 | 5.2 (8) vs 7 (n=7) | 0.29 |
| hotel.book_room | 20 | 0% | 100% | 100% | 100% | 0.40 (0.27) | 0.40 | 1.00 | 6.6 (8) vs 11 (n=11) | 0.27 |
| library.borrow_book | 20 | 100% | 100% | 100% | 100% | 0.40 (0.29) | 0.40 | 1.00 | 5.7 (11) vs 7 (n=7) | 0.29 |
| online_market.return_order | 20 | 0% | 100% | 30% | 30% | 0.00 (0.40) | 0.00 | 1.00 | 2.0 (2) vs 5 (n=5) | 0.40 |
| university.enroll_course | 20 | 0% | 100% | 100% | 100% | 0.10 (0.10) | 0.10 | 1.00 | 8.1 (12) vs 10 (n=10) | 0.10 |
| ALL | 200 | 40% | 100% | 93% | 93% | 0.34 | 0.33 | 0.98 | | |

Confusion of the LLM label against the code label on valid cases: both perform 61, LLM perform but code refuse 0, LLM refuse but code perform 3, both refuse 122.


