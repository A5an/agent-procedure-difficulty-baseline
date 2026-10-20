"""Procedure-level types, fixed before any model was run (declared in REPORT.md).
(a) fam  : action family of the requested procedure, hand-assigned from the procedure name (verb), 5 families.
(b) tree : shape of the rule tree "from the rule tree" (constraints_original; it restates the SOP text the agent sees;
           identical for the allow and the refuse variant of a case, so it does not reveal the outcome).
           size of the tree (number of checks: 1-2 / 3-4 / 5+) x (conjunction only | has an or/chain node) = 6 types.
(c) dom  : domain.
Q columns for the confirmatory MIRT (fixed loadings, 3 dims): q_deep (>=5 checks), q_branch (has or/chain), q_modify (family modify or terminate).
"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json
import pandas as pd
SOP = ZR.DATA + "/SOPBench-d2622008"
DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]
FAM = {
 "create": "apply_credit_card get_loan open_account register_vehicle schedule_test add_authorized_provider appeal_claim schedule_appointment submit_claim book_room place_room_service_order register_loyalty_member add_book borrow_book reserve_room add_review add_shipping_address add_to_cart place_order apply_financial_aid declare_minor enroll_course request_graduation",
 "terminate": "cancel_credit_card close_account cancel_test deactivate_policy cancel_reservation process_guest_checkout remove_book return_book cancel_order return_order drop_course",
 "modify": "set_account_information set_safety_box change_dl_address change_vehicle_address renew_dl renew_vehicle transfer_title update_dl_legal_name update_test_status reactivate_policy update_policy modify_reservation process_guest_checkin request_room_change update_membership exchange_product use_coupon change_major",
 "read": "get_account_owed_balance validate_vehicle_insurance get_claim_details get_policy_details get_provider_details find_booking_info show_available_rooms check_return_date get_order_details get_product_details",
 "pay": "deposit_funds exchange_foreign_currency pay_bill pay_bill_with_credit_card pay_loan transfer_funds pay_late_fee",
}
FAMOF = {p: f for f, s in FAM.items() for p in s.split()}

def leaves(c, acc):
    if c is None: return
    if c[0] in ("and", "or", "chain", "gate"):
        acc["ops"].add(c[0])
        for s in c[1]: leaves(s, acc)
    else:
        acc["n"] += 1

def build():
    rows = []
    for d in DOMAINS:
        t = json.load(open(f"{SOP}/data/{d}_tasks.json"))
        for g, lst in t.items():
            for j, x in enumerate(lst):
                acc = {"n": 0, "ops": set()}
                leaves(x["constraints_original"], acc)
                n, ops = acc["n"], acc["ops"]
                br = bool(ops & {"or", "chain", "gate"})
                size = "s12" if n <= 2 else ("s34" if n <= 4 else "s5p")
                tree = size + ("_branch" if br else "_and")
                rows.append(dict(case_id=f"{d}/{g}/{j}", procedure_id=f"{d}/{g}", dom=d, fam=FAMOF[g], tree=tree, n_checks=n,
                                 q_deep=int(n >= 5), q_branch=int(br), q_modify=int(FAMOF[g] in ("modify", "terminate"))))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = build(); df.to_csv("types.csv", index=False)
    t = pd.read_csv(ZR.REPO + "/data/sopbench/tasks.csv")
    assert set(df.case_id) == set(t.case_id) and len(df) == 830
    for c in ["fam", "tree", "dom"]: print(df[c].value_counts().to_dict())
    print(df[["q_deep", "q_branch", "q_modify"]].mean().round(2).to_dict(), df.procedure_id.nunique())
    print(pd.crosstab(df.fam, df.tree))
    print(df.groupby("procedure_id").tree.nunique().value_counts().to_dict(), "tree types per procedure")
