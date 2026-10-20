"""Step type mapping: constraint name (leaf of the SOPBench rule tree) -> step type.
Types: auth, existence, threshold, status, datetime, relation, availability, (or_group and action are structural)."""
import re

# explicit names first, then regex rules, checked in order
EXPLICIT = {
    # auth / identity
    "logged_in_user": "auth", "authenticated_admin_password": "auth", "valid_identification": "auth",
    "internal_is_admin": "auth",
    # existence / validity of an entity or a parameter value
    "internal_has_dl": "existence", "internal_has_vehicle": "existence", "internal_vehicle_registered": "existence",
    "internal_valid_room_id": "existence", "internal_valid_room_type": "existence",
    "internal_valid_room_change_reason": "existence", "internal_valid_room_service_item": "existence",
    "internal_valid_room_service_payment_method": "existence", "internal_valid_test_type": "existence",
    "has_shipping_address": "existence", "has_items_in_cart": "existence", "not_already_added_shipping_address": "existence",
    "product_exists_in_order": "relation", "has_confirmed_reservation": "relation",
    # availability / capacity
    "internal_check_foreign_currency_available": "availability", "internal_check_book_available": "availability",
    "internal_check_test_slot_available": "availability", "internal_all_slots_available_for_the_room_on_the_date": "availability",
    "internal_check_date_available_for_the_room": "availability", "course_has_capacity": "availability",
    "major_has_capacity": "availability", "room_type_available_for_dates": "availability",
    "room_type_available_for_room_change": "availability", "provider_available": "availability",
    "financial_aid_quota_available": "availability", "enough_stock": "availability",
    "database_book_not_borrowed": "availability",
    # date / time
    "valid_booking_date_pair": "datetime", "is_booking_date_within_lead_time_range": "datetime",
    "before_modification_deadline": "datetime", "after_check_in_time": "datetime", "before_check_out_time": "datetime",
    "within_room_service_hours": "datetime", "appointment_date_valid": "datetime", "before_test_date": "datetime",
    "coupon_not_expired": "datetime", "no_exam_conflict": "datetime", "no_schedule_conflict": "datetime",
    "has_overlapping_booking_for_booking": "datetime", "has_overlapping_booking_for_modification": "datetime",
    "minor_course_overlap_check": "datetime", "before_graduation_deadline": "datetime",
    # ownership / relation between two entities
    "course_enrolled_by_user": "relation", "product_bought_by_user": "relation", "user_book_borrowed": "relation",
    "user_book_not_borrowed": "relation", "provider_covers_policy": "relation", "provider_authorized": "relation",
    "provider_not_already_authorized": "relation", "minor_compatible_with_major": "relation",
    "is_dl_address_different": "relation", "is_vehicle_address_different": "relation",
    "meets_major_restriction": "relation", "meets_minor_prerequisites": "relation", "has_completed_prerequisites": "relation",
    "course_not_completed": "relation", "policy_type_valid": "relation", "meets_division_requirements": "relation",
    "major_requirements_met": "relation", "gen_ed_requirements_met": "relation", "coupon_not_already_used": "relation",
    "coupon_valid": "relation", "unique_review": "relation", "guest_already_checked_in": "relation",
    "room_key_returned": "relation", "test_scheduled": "relation", "valid_vehicle_insurance": "relation",
    "not_on_probation": "status", "internal_is_restricted": "status",
}
RULES = [
    (r"^(within|before|after)_.*(period|deadline|date|hours|time)", "datetime"),
    (r"(_period|_deadline)$", "datetime"),
    (r"^internal_check_.*exist", "existence"),
    (r"^(sufficient|minimal|maximum|minimum|enough|under_max|less_than_max|within_.*(limit|max|attempt)|has_remaining|has_min|"
     r"credits_within|credit_requirement|gpa|meets_.*(gpa|credit|income|half)|income_proof|amount_positive|"
     r"maintains_min|tuition_balance|claim_within|within_.*_limits|no_owed|no_credit_card|not_over|get_loan|pay_loan|"
     r"safety_box_eligible|has_exceeded|not_has_exceeded|not has_exceeded)", "threshold"),
    (r"(policy_(in)?active|credit_status|order_(delivered|processing)|claim_status|valid_membership|no_pending|"
     r"is_gold|is_loyalty|loyalty|drive_test_ready|test_type_is_drive|valid_residency|within_enrollment|"
     r"payment_with_loyalty)", "status"),
]


def step_type(name):
    n = re.sub(r"^not ", "", name)
    if n in EXPLICIT:
        return EXPLICIT[n]
    for pat, t in RULES:
        if re.search(pat, n):
            return t
    return "status"
