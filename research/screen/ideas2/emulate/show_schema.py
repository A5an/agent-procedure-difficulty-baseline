import json, gen, gt
for d,p in [("online_market","return_order"),("hotel","book_room"),("library","borrow_book")]:
    cs,same=gen.modal_variant(d,p)
    print(d, json.dumps(gen.skeleton(same[0]["initial_database"],fmt=True,coll=gen.collections_for(d)))[:900]); print()
