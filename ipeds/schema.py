"""Canonical schema definitions and helpers for IPEDS panel construction."""
from __future__ import annotations

import unicodedata
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _normalize_text(value: str) -> str:
    """Normalize titles for matching (lowercase, ASCII, collapse whitespace)."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("–", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class ColumnDefinition:
    canonical: str
    aliases: Tuple[str, ...]
    group: str

    def normalized_aliases(self) -> Set[str]:
        return {_normalize_text(alias) for alias in self.aliases}


def _def(canonical: str, group: str, aliases: Sequence[str]) -> ColumnDefinition:
    return ColumnDefinition(
        canonical=canonical,
        group=group,
        aliases=tuple(aliases),
    )


_COLUMN_DEFINITIONS: List[ColumnDefinition] = [
    # IC – Directory
    _def("institution_name", "ic_directory", ["Institution (entity) name"]),
    _def("institution_name_alias", "ic_directory", ["Institution name alias"]),
    _def("street_address_or_po_box", "ic_directory", ["Street address or post office box"]),
    _def("city_location_of_institution", "ic_directory", ["City location of institution"]),
    _def("state_abbreviation", "ic_directory", ["State abbreviation"]),
    _def("zip_code", "ic_directory", ["ZIP code"]),
    _def("county_name", "ic_directory", ["County name"]),
    _def(
        "state_and_118th_congressional_district_id",
        "ic_directory",
        ["State and 118TH Congressional District ID"],
    ),
    _def("longitude_location_of_institution", "ic_directory", ["Longitude location of institution"]),
    _def("latitude_location_of_institution", "ic_directory", ["Latitude location of institution"]),
    _def("institution_open_to_public", "ic_directory", ["Institution open to the general public"]),
    _def("status_of_institution", "ic_directory", ["Status of institution"]),
    _def("unitid_for_merged_schools", "ic_directory", ["UNITID for merged schools"]),
    _def("year_institution_deleted", "ic_directory", ["Year institution was deleted from IPEDS"]),
    _def("institution_active_current_year", "ic_directory", ["Institution is active in current year"]),
    _def("primarily_postsecondary_indicator", "ic_directory", ["Primarily postsecondary indicator"]),
    _def("postsecondary_institution_indicator", "ic_directory", ["Postsecondary institution indicator"]),
    _def(
        "postsecondary_and_title_iv_indicator",
        "ic_directory",
        ["Postsecondary and Title IV institution indicator"],
    ),
    _def("natural_disaster_identification", "ic_directory", ["Natural Disaster identification"]),
    _def(
        "status_ic_component_migrated",
        "ic_directory",
        ["Status of IC component when institution was migrated"],
    ),
    _def(
        "response_status_ic",
        "ic_directory",
        ["Response status - Institutional characteristics component"],
    ),
    _def("imputation_method_ic", "ic_directory", ["Type of imputation method Institutional Characteristics"]),
    _def("revision_status_ic", "ic_directory", ["Revision status - Institutional Characteristics"]),
    _def(
        "multi_institution_or_campus_org",
        "ic_directory",
        ["Multi-institution or multi-campus organization"],
    ),
    _def(
        "multi_institution_org_id",
        "ic_directory",
        ["Identification number of multi-institution or multi-campus organization"],
    ),
    _def(
        "multi_institution_org_name",
        "ic_directory",
        ["Name of multi-institution or multi-campus organization"],
    ),
    _def("employer_identification_number", "ic_directory", ["Employer Identification Number"]),
    _def("unique_entity_identifier", "ic_directory", ["Unique Entity Identifier (UEI) Numbers"]),
    _def("ope_id_number", "ic_directory", ["Office of Postsecondary Education (OPE) ID Number"]),
    _def(
        "ope_title_iv_indicator_code",
        "ic_directory",
        ["OPE Title IV eligibility indicator code"],
    ),
    _def("core_based_statistical_area", "ic_directory", ["Core Based Statistical Area (CBSA)"]),
    _def("cbsa_type", "ic_directory", ["CBSA Type Metropolitan or Micropolitan"]),
    _def("combined_statistical_area", "ic_directory", ["Combined Statistical Area (CSA)"]),
    _def("fips_county_code", "ic_directory", ["Fips County code", "FIPS county code"]),
    _def("fips_state_code", "ic_directory", ["FIPS state code"]),
    _def(
        "bea_region",
        "ic_directory",
        ["Bureau of Economic Analysis (BEA) regions"],
    ),
    # IC – Charges & Cost of Attendance (AY current)
    _def("published_tuition_and_fees_2023_24", "ic_cst", ["Published tuition and fees 2023-24"]),
    _def(
        "published_out_of_state_tuition_and_fees_2023_24",
        "ic_cst",
        ["Published out-of-state tuition and fees 2023-24"],
    ),
    _def("books_and_supplies_2023_24", "ic_cst", ["Books and supplies 2023-24"]),
    _def(
        "total_price_in_district_on_campus_2023_24",
        "ic_cst",
        ["Total price for in-district students living on campus 2023-24"],
    ),
    _def(
        "total_price_in_district_off_campus_2023_24",
        "ic_cst",
        ["Total price for in-district students living off campus (not with family) 2023-24"],
    ),
    _def(
        "total_price_in_state_on_campus_2023_24",
        "ic_cst",
        ["Total price for in-state students living on campus 2023-24"],
    ),
    _def(
        "total_price_in_state_off_campus_2023_24",
        "ic_cst",
        ["Total price for in-state students living off campus (not with family) 2023-24"],
    ),
    _def(
        "total_price_out_of_state_on_campus_2023_24",
        "ic_cst",
        ["Total price for out-of-state students living on campus 2023-24"],
    ),
    _def(
        "total_price_out_of_state_off_campus_2023_24",
        "ic_cst",
        ["Total price for out-of-state students living off campus (not with family) 2023-24"],
    ),
    _def(
        "off_campus_with_family_expenses_2023_24",
        "ic_cst",
        ["Off campus (with family), other expenses 2023-24"],
    ),
    _def(
        "off_campus_not_with_family_expenses_2023_24",
        "ic_cst",
        ["Off campus (not with family), other expenses 2023-24"],
    ),
    # Finance – Revenues & return
    _def(
        "rev_tuition_fees_net",
        "finance_revenue",
        [
            "Tuition and fees (net of allowance reported in Part C, line 08)",
            "Tuition and fees (net of allowances)",
            "Net tuition and fees",
        ],
    ),
    _def("rev_federal_appropriations", "finance_revenue", ["Federal appropriations"]),
    _def("rev_state_appropriations", "finance_revenue", ["State appropriations"]),
    _def("rev_local_appropriations", "finance_revenue", ["Local appropriations"]),
    _def(
        "rev_federal_grants_contracts",
        "finance_revenue",
        [
            "Federal grants and contracts (Do not include FDSL)",
            "Federal grants and contracts",
        ],
    ),
    _def("rev_state_grants_contracts", "finance_revenue", ["State grants and contracts"]),
    _def("rev_local_grants_contracts", "finance_revenue", ["Local government grants and contracts"]),
    _def(
        "rev_private_gifts_grants_contracts",
        "finance_revenue",
        [
            "Private gifts, grants and contracts",
            "Private gifts, grants, and contracts",
        ],
    ),
    _def("rev_private_gifts", "finance_revenue", ["Private gifts"]),
    _def("rev_private_grants_contracts", "finance_revenue", ["Private grants and contracts"]),
    _def(
        "rev_contributions_affiliated_entities",
        "finance_revenue",
        ["Contributions from affiliated entities"],
    ),
    _def("rev_investment_return", "finance_revenue", ["Investment return"]),
    _def(
        "rev_sales_services_education",
        "finance_revenue",
        ["Sales and services of educational activities"],
    ),
    _def(
        "rev_sales_services_auxiliary_net",
        "finance_revenue",
        [
            "Sales and services of auxiliary enterprises (net of allowance reported in Part C, line 09)",
            "Sales and services of auxiliary enterprises (net)",
        ],
    ),
    _def("rev_hospital", "finance_revenue", ["Hospital revenue"]),
    _def("rev_independent_operations", "finance_revenue", ["Independent operations revenue"]),
    _def("rev_other", "finance_revenue", ["Other revenue", "Other revenue (calculated)"]),
    _def(
        "rev_total_and_investment_return",
        "finance_revenue",
        ["Total revenues and investment return"],
    ),
    _def("rev_net_assets_released", "finance_revenue", ["Net assets released from restriction"]),
    _def(
        "rev_net_total_after_restriction",
        "finance_revenue",
        ["Net total revenues, after assets released from restriction"],
    ),
    # Finance – Discounts & scholarships
    _def("disc_pell_grants", "finance_discounts", ["Pell grants (federal)"]),
    _def("disc_federal_other", "finance_discounts", ["Other federal grants"]),
    _def("disc_state_grants", "finance_discounts", ["Grants by state government"]),
    _def("disc_local_grants", "finance_discounts", ["Grants by local government"]),
    _def("disc_inst_restricted", "finance_discounts", ["Institutional grants (restricted)"]),
    _def("disc_inst_unrestricted", "finance_discounts", ["Institutional grants (unrestricted)"]),
    _def(
        "disc_total_revenue_scholarships",
        "finance_discounts",
        ["Total revenue that funds scholarships and fellowships"],
    ),
    _def(
        "disc_tuition_allowances",
        "finance_discounts",
        ["Discounts and allowances applied to tuition and fees"],
    ),
    _def(
        "disc_auxiliary_allowances",
        "finance_discounts",
        ["Discounts and allowances applied to auxiliary enterprise revenues"],
    ),
    _def("disc_total_allowances", "finance_discounts", ["Total discounts and allowances"]),
    # Finance – Expenses (current year totals + salaries)
    _def("exp_instruction_total", "finance_expenses", ["Instruction — Current year total"]),
    _def("exp_instruction_salaries", "finance_expenses", ["Instruction — Salaries and wages"]),
    _def("exp_research_total", "finance_expenses", ["Research — Current year total"]),
    _def("exp_research_salaries", "finance_expenses", ["Research — Salaries and wages"]),
    _def("exp_public_service_total", "finance_expenses", ["Public service — Current year total"]),
    _def("exp_public_service_salaries", "finance_expenses", ["Public service — Salaries and wages"]),
    _def("exp_academic_support_total", "finance_expenses", ["Academic support — Current year total"]),
    _def("exp_academic_support_salaries", "finance_expenses", ["Academic support — Salaries and wages"]),
    _def("exp_student_services_total", "finance_expenses", ["Student services — Current year total"]),
    _def("exp_student_services_salaries", "finance_expenses", ["Student services — Salaries and wages"]),
    _def("exp_institutional_support_total", "finance_expenses", ["Institutional support — Current year total"]),
    _def("exp_institutional_support_salaries", "finance_expenses", ["Institutional support — Salaries and wages"]),
    _def("exp_auxiliary_total", "finance_expenses", ["Auxiliary enterprises — Current year total"]),
    _def("exp_auxiliary_salaries", "finance_expenses", ["Auxiliary enterprises — Salaries and wages"]),
    _def(
        "exp_net_grant_aid_total",
        "finance_expenses",
        ["Net grant aid to students, net of discount/allowances — Current year total"],
    ),
    _def("exp_hospital_total", "finance_expenses", ["Hospital services — Current year total"]),
    _def("exp_independent_operations_total", "finance_expenses", ["Independent operations — Current year total"]),
    _def(
        "exp_other_deductions_total",
        "finance_expenses",
        ["Other Functional Expenses and deductions — Current year total"],
    ),
    _def("exp_total_all", "finance_expenses", ["Total expenses and Deductions"]),
    # Finance – Endowment
    _def(
        "endowment_value_begin",
        "finance_endowment",
        ["Value of endowment net assets at the beginning of the fiscal year"],
    ),
    _def(
        "endowment_value_end",
        "finance_endowment",
        ["Value of endowment net assets at the end of the fiscal year"],
    ),
    _def(
        "endowment_change",
        "finance_endowment",
        ["Change in value of endowment net assets"],
    ),
    _def("endowment_new_gifts", "finance_endowment", ["New gifts and additions"]),
    _def("endowment_net_invest_return", "finance_endowment", ["Endowment net investment return"]),
    _def(
        "endowment_spending_distribution",
        "finance_endowment",
        ["Spending distribution for current use"],
    ),
    _def("endowment_other", "finance_endowment", ["Other"]),
    # SFA – Average net price (AY 2022-23)
    _def(
        "sfa_avg_net_price_all_2022_23",
        "sfa_net_price",
        ["Average net price for full-time, first-time degree/certificate-seeking undergraduates awarded Title IV federal financial aid, 2022-23"],
    ),
    _def(
        "sfa_avg_net_price_0_30k_2022_23",
        "sfa_net_price",
        ["Average net price (income $0–$30,000) awarded Title IV federal financial aid, 2022-23"],
    ),
    _def(
        "sfa_avg_net_price_30_48k_2022_23",
        "sfa_net_price",
        ["Average net price (income $30,001–$48,000) awarded Title IV federal financial aid, 2022-23"],
    ),
    _def(
        "sfa_avg_net_price_48_75k_2022_23",
        "sfa_net_price",
        ["Average net price (income $48,001–$75,000) awarded Title IV federal financial aid, 2022-23"],
    ),
    _def(
        "sfa_avg_net_price_75_110k_2022_23",
        "sfa_net_price",
        ["Average net price (income $75,001–$110,000) awarded Title IV federal financial aid, 2022-23"],
    ),
    _def(
        "sfa_avg_net_price_gt_110k_2022_23",
        "sfa_net_price",
        ["Average net price (income >$110,000) awarded Title IV federal financial aid, 2022-23"],
    ),
    _def(
        "sfa_avg_net_price_pub_instate_2022_23",
        "sfa_net_price",
        ["Average net price for students paying the in-state or in-district tuition rate (public institutions), 2022-23"],
    ),
    # E12 – 12-month enrollment
    _def("e12_total_unduplicated_headcount", "e12", ["Total 12-month unduplicated headcount"]),
    _def("e12_fte_enrollment", "e12", ["12-month full-time equivalent enrollment"]),
    _def(
        "e12_full_time_unduplicated_headcount",
        "e12",
        ["Full-time 12-month unduplicated headcount"],
    ),
    _def(
        "e12_part_time_unduplicated_headcount",
        "e12",
        ["Part-time 12-month unduplicated headcount"],
    ),
    _def(
        "e12_first_time_degree_cert_ug_headcount",
        "e12",
        ["First-time degree/certificate-seeking undergraduate 12-month unduplicated headcount"],
    ),
    _def(
        "e12_undergraduate_unduplicated_headcount",
        "e12",
        ["Undergraduate 12-month unduplicated headcount"],
    ),
    _def("e12_graduate_unduplicated_headcount", "e12", ["Graduate 12-month unduplicated headcount"]),
    # EF – Fall enrollment
    _def("ef_institution_size_category", "ef", ["Institution size category"]),
    _def("ef_total_fall_enrollment", "ef", ["Total fall enrollment"]),
    _def("ef_part_time_enrollment", "ef", ["Part-time fall enrollment"]),
    _def("ef_fte_fall_enrollment", "ef", ["Full-time equivalent fall enrollment"]),
    _def("ef_ug_full_time", "ef", ["Undergraduate fall enrollment — full-time"]),
    _def("ef_ug_part_time", "ef", ["Undergraduate fall enrollment — part-time"]),
    _def("ef_gr_part_time", "ef", ["Graduate fall enrollment — part-time"]),
    # HR – Instructional staff salaries
    _def(
        "hr_salary_all_ranks_9mo",
        "hr",
        ["Average salary equated to 9 months of full-time instructional staff — all ranks"],
    ),
    _def(
        "hr_salary_professors_9mo",
        "hr",
        ["Average salary equated to 9 months of full-time instructional staff — professors"],
    ),
    _def(
        "hr_salary_assoc_prof_9mo",
        "hr",
        ["Average salary equated to 9 months of full-time instructional staff — associate professors"],
    ),
    _def(
        "hr_salary_asst_prof_9mo",
        "hr",
        ["Average salary equated to 9 months of full-time instructional staff — assistant professors"],
    ),
    _def(
        "hr_salary_instructors_9mo",
        "hr",
        ["Average salary equated to 9 months of full-time instructional staff — instructors"],
    ),
    # Flags appended later
    _def("finance_form_type", "flags", ["finance_form_type"]),
    _def("finance_parent_child_indicator", "flags", ["finance_parent_child_indicator"]),
    _def("finance_accounting_standard", "flags", ["finance_accounting_standard"]),
]


CANONICAL_ORDER: List[str] = [definition.canonical for definition in _COLUMN_DEFINITIONS]

_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for definition in _COLUMN_DEFINITIONS:
    for alias in definition.normalized_aliases():
        _ALIAS_TO_CANONICAL[alias] = definition.canonical


COLUMN_DEFINITIONS: Dict[str, ColumnDefinition] = {
    definition.canonical: definition for definition in _COLUMN_DEFINITIONS
}


GROUP_YEAR_OFFSETS: Dict[str, int] = {
    "ic_directory": 0,
    "ic_cst": 0,
    "ef": 0,
    "hr": 0,
    "flags": 0,
    "sfa_net_price": -1,
    "e12": -1,
    "finance_revenue": -1,
    "finance_discounts": -1,
    "finance_expenses": -1,
    "finance_endowment": -1,
}


def canonical_for_title(title: str) -> Optional[str]:
    """Return canonical column name for a varTitle or synonym."""
    normalized = _normalize_text(title)
    return _ALIAS_TO_CANONICAL.get(normalized)


def aliases_for_canonical(canonical: str) -> List[str]:
    definition = COLUMN_DEFINITIONS.get(canonical)
    if not definition:
        return []
    return list(definition.aliases)


def normalized_aliases_for_canonical(canonical: str) -> Set[str]:
    definition = COLUMN_DEFINITIONS.get(canonical)
    if not definition:
        return set()
    return definition.normalized_aliases()


def group_for_canonical(canonical: str) -> Optional[str]:
    definition = COLUMN_DEFINITIONS.get(canonical)
    if not definition:
        return None
    return definition.group


def panel_year_offset(canonical: str) -> int:
    group = group_for_canonical(canonical)
    if not group:
        return 0
    return GROUP_YEAR_OFFSETS.get(group, 0)


def normalize_title(title: str) -> str:
    """Expose normalization helper for external modules."""
    return _normalize_text(title)


def canonical_columns() -> List[str]:
    """Return canonical column order."""
    return CANONICAL_ORDER.copy()


def is_finance_canonical(canonical: str) -> bool:
    group = group_for_canonical(canonical)
    return bool(group and group.startswith("finance_"))


def finance_form_from_table(table_name: str) -> Optional[str]:
    """Infer finance form (F1/F2/F3) from a table name."""
    if not table_name:
        return None
    table_name = table_name.upper()
    if table_name.startswith("F1"):
        return "F1"
    if table_name.startswith("F2"):
        return "F2"
    if table_name.startswith("F3"):
        return "F3"
    return None
