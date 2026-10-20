import json, gt, collections
SOP=gt.SOP
sel=[("bank","pay_loan"),("bank","apply_credit_card"),("dmv","schedule_test"),("dmv","renew_vehicle"),("healthcare","schedule_appointment"),("healthcare","submit_claim"),("hotel","book_room"),("library","borrow_book"),("online_market","return_order"),("university","enroll_course")]
for dom,p in sel:
    cs=json.load(open(f"{SOP}/data/{dom}_tasks.json"))[p]
    ag=0
    for c in cs:
        lab,_=gt.leaf_results(dom,c,goal=p); ag+= lab==c['action_should_succeed']
    ins,tools=gt.policy_prompt(dom,cs[0],goal=p)
    print(dom,p,len(cs),'agree',ag,'perform',sum(c['action_should_succeed'] for c in cs),'dbchars',[len(json.dumps(c['initial_database'])) for c in cs][:3],'prompt chars',len(ins),'tools chars',len(json.dumps(tools)))
