Best of the 25 real agents: o4-mini-high-mode_fc overall 0.753
Median of the 25: gpt-4o-mini-mode_fc 0.417
Always-refuse baseline (call nothing): success = share of refuse cases = 0.653
results_A.json minus best agent: +0.017 [-0.012, +0.047] (paired bootstrap over cases)
results_A.json minus median agent: +0.353 [+0.312, +0.392] (paired bootstrap over cases)
   perform: compiled 0.431, best agent 0.559
   refuse: compiled 0.950, best agent 0.856
results_B.json minus best agent: +0.011 [-0.019, +0.040] (paired bootstrap over cases)
results_B.json minus median agent: +0.347 [+0.310, +0.384] (paired bootstrap over cases)
   perform: compiled 0.406, best agent 0.559
   refuse: compiled 0.954, best agent 0.856
results_AC.json minus best agent: +0.110 [+0.084, +0.135] (paired bootstrap over cases)
results_AC.json minus median agent: +0.446 [+0.408, +0.483] (paired bootstrap over cases)
   perform: compiled 0.691, best agent 0.559
   refuse: compiled 0.954, best agent 0.856
results_BC.json minus best agent: +0.083 [+0.059, +0.110] (paired bootstrap over cases)
results_BC.json minus median agent: +0.419 [+0.381, +0.458] (paired bootstrap over cases)
   perform: compiled 0.615, best agent 0.559
   refuse: compiled 0.954, best agent 0.856
Procedures where compiled(C,A) is below the agent mean: bank/get_account_owed_balance (0.5 vs 0.62), bank/pay_bill_with_credit_card (0.2 vs 0.288), healthcare/get_provider_details (0.0 vs 0.36), online_market/get_product_details (0.0 vs 0.32)
Procedures with compiled(C,A) success below 0.6: bank/cancel_credit_card n=17 (0.471), bank/exchange_foreign_currency n=4 (0.5), bank/get_account_owed_balance n=2 (0.5), bank/get_loan n=12 (0.583), bank/open_account n=5 (0.4), bank/pay_bill_with_credit_card n=5 (0.2), healthcare/get_provider_details n=1 (0.0), hotel/find_booking_info n=1 (0.0), hotel/process_guest_checkin n=7 (0.571), hotel/show_available_rooms n=1 (0.0), library/remove_book n=2 (0.5), library/update_membership n=2 (0.5), online_market/get_product_details n=1 (0.0)
Procedures with all cases passed (C,A): 36 of 70; compiled success >=0.8: 42