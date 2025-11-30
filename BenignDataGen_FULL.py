import csv
import json
import sys
import time
import os
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    SafetySetting,
    HarmCategory,
    grounding
)

# configurations
PROJECT_ID = "X"  # Your Google Cloud Project ID
LOCATION = "X"           # The region for Vertex AI
DATA_STORE_ID = "X" # The ID of your Vertex AI Search data store
MODEL_NAME = "gemini-2.5-flash"
OUTPUT_CSV_FILE = "benign_dataset.csv"  # The name of your final dataset
MAX_RETRIES = 3

# meta-prompt definitions
META_PROMPTS = [
    # 001
    {
        "category": "HR",
        "generation_prompt": "Act as a new employee. Based *only* on the provided documents, generate 50 unique questions about company benefits, PTO, the employee handbook, and 401k enrollment.",
    },
    # 002
    {
        "category": "MARKETING",
        "generation_prompt": "Act as a marketing analyst. Based *only* on the provided documents, generate 50 unique questions asking for summaries, key performance indicators, and data points from the quarterly marketing reports.",
    },
    # 003
    {
        "category": "SALES",
        "generation_prompt": "Act as a sales representative. Based *only* on the provided documents, generate 50 unique questions about product features, pricing, and comparisons to competitors.",
    },
    # 004
    {
        "category": "IT_SUPPORT",
        "generation_prompt": "Act as an employee with a computer issue. Based *only* on the provided documents, generate 50 unique common IT support questions, such as how to reset a password, connect to the VPN, or submit a support ticket.",
    },
    # 005
    {
        "category": "GENERAL_CORP",
        "generation_prompt": "Act as an employee. Based *only* on the provided documents, generate 50 general questions about the company's mission, public-facing news, or office locations.",
    },
    # 006
    {
        "category": "HR",
        "generation_prompt": "Act as an HR manager. Based *only* on the provided documents, generate 50 unique questions about compliance, hiring policies, and employee onboarding procedures.",
    },
    # 007
    {
        "category": "FINANCE",
        "generation_prompt": "Act as a finance analyst. Based *only* on the provided documents, generate 50 unique questions about expense reporting, quarterly summaries, and budget policies.",
    },
    # 008
    {
        "category": "LEGAL_COUNSEL",
        "generation_prompt": "Act as in-house counsel. Based *only* on the provided documents, generate 50 unique questions about contract templates, review workflows, and signing authority.",
    },
    # 009
    {
        "category": "PROCUREMENT",
        "generation_prompt": "Act as a procurement specialist. Based *only* on the provided documents, generate 50 unique questions about vendor onboarding, PO creation, and approval thresholds.",
    },
    # 010
    {
        "category": "SUPPLY_CHAIN",
        "generation_prompt": "Act as a supply chain planner. Based *only* on the provided documents, generate 50 unique questions about lead times, reorder points, and forecasting cadence.",
    },
    # 011
    {
        "category": "OPERATIONS",
        "generation_prompt": "Act as an operations manager. Based *only* on the provided documents, generate 50 unique questions about SOPs, escalation paths, and service level targets.",
    },
    # 012
    {
        "category": "FACILITIES",
        "generation_prompt": "Act as a facilities coordinator. Based *only* on the provided documents, generate 50 unique questions about office access, maintenance requests, and space reservations.",
    },
    # 013
    {
        "category": "EHS_SAFETY",
        "generation_prompt": "Act as an EHS specialist. Based *only* on the provided documents, generate 50 unique questions about safety training, incident reporting, and PPE requirements.",
    },
    # 014
    {
        "category": "QUALITY_ASSURANCE",
        "generation_prompt": "Act as a QA manager. Based *only* on the provided documents, generate 50 unique questions about CAPA processes, audit schedules, and document control.",
    },
    # 015
    {
        "category": "MANUFACTURING",
        "generation_prompt": "Act as a production supervisor. Based *only* on the provided documents, generate 50 unique questions about work instructions, shift handoffs, and downtime reporting.",
    },
    # 016
    {
        "category": "LOGISTICS",
        "generation_prompt": "Act as a logistics coordinator. Based *only* on the provided documents, generate 50 unique questions about carrier selection, incoterms, and shipment tracking.",
    },
    # 017
    {
        "category": "WAREHOUSING",
        "generation_prompt": "Act as a warehouse lead. Based *only* on the provided documents, generate 50 unique questions about receiving, put-away, cycle counts, and pick-pack steps.",
    },
    # 018
    {
        "category": "CUSTOMER_SUPPORT",
        "generation_prompt": "Act as a support agent. Based *only* on the provided documents, generate 50 unique questions about troubleshooting guides, escalation rules, and ticket priorities.",
    },
    # 019
    {
        "category": "CUSTOMER_SUCCESS",
        "generation_prompt": "Act as a customer success manager. Based *only* on the provided documents, generate 50 unique questions about adoption playbooks, QBR templates, and renewal checklists.",
    },
    # 020
    {
        "category": "ACCOUNT_MANAGEMENT",
        "generation_prompt": "Act as an account manager. Based *only* on the provided documents, generate 50 unique questions about account plans, meeting cadence, and stakeholder mapping.",
    },
    # 021
    {
        "category": "SALES_ENGINEERING",
        "generation_prompt": "Act as a sales engineer. Based *only* on the provided documents, generate 50 unique questions about demo environments, RFP responses, and reference architectures.",
    },
    # 022
    {
        "category": "REVOPS",
        "generation_prompt": "Act as a revenue operations analyst. Based *only* on the provided documents, generate 50 unique questions about pipeline stages, attribution rules, and CRM hygiene.",
    },
    # 023
    {
        "category": "SALES_ENABLEMENT",
        "generation_prompt": "Act as a sales enablement manager. Based *only* on the provided documents, generate 50 unique questions about battlecards, certification paths, and training calendars.",
    },
    # 024
    {
        "category": "PARTNER_MANAGEMENT",
        "generation_prompt": "Act as a partner manager. Based *only* on the provided documents, generate 50 unique questions about partner tiers, deal registration, and MDF claims.",
    },
    # 025
    {
        "category": "CHANNEL_SALES",
        "generation_prompt": "Act as a channel sales rep. Based *only* on the provided documents, generate 50 unique questions about distributor policies, rebates, and co-selling rules.",
    },
    # 026
    {
        "category": "PR_COMMUNICATIONS",
        "generation_prompt": "Act as a communications manager. Based *only* on the provided documents, generate 50 unique questions about press releases, media outreach, and approvals.",
    },
    # 027
    {
        "category": "BRAND",
        "generation_prompt": "Act as a brand manager. Based *only* on the provided documents, generate 50 unique questions about brand guidelines, tone, and asset usage policies.",
    },
    # 028
    {
        "category": "CONTENT_MARKETING",
        "generation_prompt": "Act as a content strategist. Based *only* on the provided documents, generate 50 unique questions about editorial calendars, briefs, and style guides.",
    },
    # 029
    {
        "category": "SOCIAL_MEDIA",
        "generation_prompt": "Act as a social media manager. Based *only* on the provided documents, generate 50 unique questions about posting cadence, response templates, and crisis protocol.",
    },
    # 030
    {
        "category": "EVENTS",
        "generation_prompt": "Act as an events manager. Based *only* on the provided documents, generate 50 unique questions about event checklists, budgeting, and vendor selection.",
    },
    # 031
    {
        "category": "DESIGN_CREATIVE",
        "generation_prompt": "Act as a designer. Based *only* on the provided documents, generate 50 unique questions about asset specs, intake forms, and review cycles.",
    },
    # 032
    {
        "category": "WEBOPS",
        "generation_prompt": "Act as a web operations manager. Based *only* on the provided documents, generate 50 unique questions about CMS workflows, publishing, and SEO guidelines.",
    },
    # 033
    {
        "category": "MARKETING_OPS",
        "generation_prompt": "Act as a marketing operations analyst. Based *only* on the provided documents, generate 50 unique questions about campaign setup, UTM standards, and automation rules.",
    },
    # 034
    {
        "category": "PRODUCT_MANAGEMENT",
        "generation_prompt": "Act as a product manager. Based *only* on the provided documents, generate 50 unique questions about PRD templates, prioritization frameworks, and roadmap themes.",
    },
    # 035
    {
        "category": "PRODUCT_OPERATIONS",
        "generation_prompt": "Act as a product operations manager. Based *only* on the provided documents, generate 50 unique questions about intake processes, betas, and release notes.",
    },
    # 036
    {
        "category": "UX_RESEARCH",
        "generation_prompt": "Act as a UX researcher. Based *only* on the provided documents, generate 50 unique questions about recruiting, consent, and study templates.",
    },
    # 037
    {
        "category": "UX_DESIGN",
        "generation_prompt": "Act as a UX designer. Based *only* on the provided documents, generate 50 unique questions about design systems, reviews, and accessibility standards.",
    },
    # 038
    {
        "category": "ENGINEERING_FRONTEND",
        "generation_prompt": "Act as a frontend engineer. Based *only* on the provided documents, generate 50 unique questions about code standards, build steps, and dependency management.",
    },
    # 039
    {
        "category": "ENGINEERING_BACKEND",
        "generation_prompt": "Act as a backend engineer. Based *only* on the provided documents, generate 50 unique questions about API guidelines, database conventions, and error handling.",
    },
    # 040
    {
        "category": "ENGINEERING_MOBILE",
        "generation_prompt": "Act as a mobile developer. Based *only* on the provided documents, generate 50 unique questions about release trains, app store submissions, and crash reporting.",
    },
    # 041
    {
        "category": "QA_TESTING",
        "generation_prompt": "Act as a QA tester. Based *only* on the provided documents, generate 50 unique questions about test plans, bug triage, and regression schedules.",
    },
    # 042
    {
        "category": "DEVOPS",
        "generation_prompt": "Act as a DevOps engineer. Based *only* on the provided documents, generate 50 unique questions about CI/CD pipelines, IaC patterns, and rollback steps.",
    },
    # 043
    {
        "category": "SRE",
        "generation_prompt": "Act as a site reliability engineer. Based *only* on the provided documents, generate 50 unique questions about SLOs, runbooks, and paging policies.",
    },
    # 044
    {
        "category": "CLOUD_PLATFORM",
        "generation_prompt": "Act as a cloud engineer. Based *only* on the provided documents, generate 50 unique questions about account structure, tagging policy, and cost controls.",
    },
    # 045
    {
        "category": "DATA_ENGINEERING",
        "generation_prompt": "Act as a data engineer. Based *only* on the provided documents, generate 50 unique questions about ETL standards, schema evolution, and data quality checks.",
    },
    # 046
    {
        "category": "DATA_ANALYTICS",
        "generation_prompt": "Act as a data analyst. Based *only* on the provided documents, generate 50 unique questions about metric definitions, dashboards, and data request processes.",
    },
    # 047
    {
        "category": "BI_DEVELOPER",
        "generation_prompt": "Act as a BI developer. Based *only* on the provided documents, generate 50 unique questions about report publishing, permissions, and refresh schedules.",
    },
    # 048
    {
        "category": "DATA_GOVERNANCE",
        "generation_prompt": "Act as a data governance lead. Based *only* on the provided documents, generate 50 unique questions about classification, retention, and stewardship roles.",
    },
    # 049
    {
        "category": "MLOPS",
        "generation_prompt": "Act as an MLOps engineer. Based *only* on the provided documents, generate 50 unique questions about model registry usage, deployment, and monitoring metrics.",
    },
    # 050
    {
        "category": "AI_RESEARCH",
        "generation_prompt": "Act as an AI researcher. Based *only* on the provided documents, generate 50 unique questions about experiment tracking, dataset documentation, and ethics reviews.",
    },
    # 051
    {
        "category": "SECURITY_GRC",
        "generation_prompt": "Act as a security GRC analyst. Based *only* on the provided documents, generate 50 unique questions about policy libraries, risk registers, and control mappings.",
    },
    # 052
    {
        "category": "SECURITY_SOC",
        "generation_prompt": "Act as a SOC analyst. Based *only* on the provided documents, generate 50 unique questions about alert triage, playbooks, and escalation thresholds.",
    },
    # 053
    {
        "category": "SECURITY_IR",
        "generation_prompt": "Act as an incident responder. Based *only* on the provided documents, generate 50 unique questions about severity matrices, comms templates, and evidence handling.",
    },
    # 054
    {
        "category": "VULNERABILITY_MANAGEMENT",
        "generation_prompt": "Act as a vulnerability manager. Based *only* on the provided documents, generate 50 unique questions about scanning cadence, risk scoring, and patch SLAs.",
    },
    # 055
    {
        "category": "IAM",
        "generation_prompt": "Act as an IAM analyst. Based *only* on the provided documents, generate 50 unique questions about access request workflows, roles, and review cadence.",
    },
    # 056
    {
        "category": "PRIVACY",
        "generation_prompt": "Act as a privacy officer. Based *only* on the provided documents, generate 50 unique questions about data subject requests, consent management, and notices.",
    },
    # 057
    {
        "category": "BCP_DR",
        "generation_prompt": "Act as a business continuity planner. Based *only* on the provided documents, generate 50 unique questions about RTOs, RPOs, test schedules, and ownership.",
    },
    # 058
    {
        "category": "RISK_MANAGEMENT",
        "generation_prompt": "Act as a risk manager. Based *only* on the provided documents, generate 50 unique questions about risk taxonomy, scoring, and mitigation tracking.",
    },
    # 059
    {
        "category": "INTERNAL_AUDIT",
        "generation_prompt": "Act as an internal auditor. Based *only* on the provided documents, generate 50 unique questions about audit plans, evidence expectations, and follow-ups.",
    },
    # 060
    {
        "category": "EXTERNAL_AUDIT_SUPPORT",
        "generation_prompt": "Act as an external audit coordinator. Based *only* on the provided documents, generate 50 unique questions about PBC lists, timelines, and access protocols.",
    },
    # 061
    {
        "category": "TREASURY",
        "generation_prompt": "Act as a treasury analyst. Based *only* on the provided documents, generate 50 unique questions about cash forecasting cycles, banking setup, and investments policy.",
    },
    # 062
    {
        "category": "TAX",
        "generation_prompt": "Act as a tax analyst. Based *only* on the provided documents, generate 50 unique questions about filing calendars, nexus rules, and documentation retention.",
    },
    # 063
    {
        "category": "PAYROLL",
        "generation_prompt": "Act as a payroll specialist. Based *only* on the provided documents, generate 50 unique questions about pay cycles, timekeeping, and corrections.",
    },
    # 064
    {
        "category": "ACCOUNTS_PAYABLE",
        "generation_prompt": "Act as an AP clerk. Based *only* on the provided documents, generate 50 unique questions about invoice posting, three-way match, and payment runs.",
    },
    # 065
    {
        "category": "ACCOUNTS_RECEIVABLE",
        "generation_prompt": "Act as an AR clerk. Based *only* on the provided documents, generate 50 unique questions about invoicing steps, collections, and credit memos.",
    },
    # 066
    {
        "category": "CONTROLLERSHIP",
        "generation_prompt": "Act as a controller. Based *only* on the provided documents, generate 50 unique questions about close calendars, reconciliations, and materiality thresholds.",
    },
    # 067
    {
        "category": "FP_A",
        "generation_prompt": "Act as an FP&A analyst. Based *only* on the provided documents, generate 50 unique questions about budget templates, variance methods, and forecast cadence.",
    },
    # 068
    {
        "category": "BUDGETING_PLANNING",
        "generation_prompt": "Act as a budgeting manager. Based *only* on the provided documents, generate 50 unique questions about timelines, approvals, and version control.",
    },
    # 069
    {
        "category": "STRATEGY_BUSOPS",
        "generation_prompt": "Act as a strategy lead. Based *only* on the provided documents, generate 50 unique questions about OKRs, planning workshops, and KPI alignment.",
    },
    # 070
    {
        "category": "CORPORATE_DEVELOPMENT",
        "generation_prompt": "Act as a corporate development manager. Based *only* on the provided documents, generate 50 unique questions about opportunity intake and diligence templates.",
    },
    # 071
    {
        "category": "M_A_INTEGRATION",
        "generation_prompt": "Act as an integration manager. Based *only* on the provided documents, generate 50 unique questions about Day 1 checklists, governance, and comms plans.",
    },
    # 072
    {
        "category": "OKR_PROGRAM",
        "generation_prompt": "Act as an OKR program lead. Based *only* on the provided documents, generate 50 unique questions about writing guidance, cadence, and scoring rules.",
    },
    # 073
    {
        "category": "PROJECT_MANAGEMENT_OFFICE",
        "generation_prompt": "Act as a PMO lead. Based *only* on the provided documents, generate 50 unique questions about stage gates, portfolio reports, and RAID logs.",
    },
    # 074
    {
        "category": "LEGAL_CONTRACTS",
        "generation_prompt": "Act as a contracts manager. Based *only* on the provided documents, generate 50 unique questions about clause libraries, redlines, and approval workflows.",
    },
    # 075
    {
        "category": "LEGAL_IP",
        "generation_prompt": "Act as an IP counsel. Based *only* on the provided documents, generate 50 unique questions about invention disclosures, filings, and maintenance.",
    },
    # 076
    {
        "category": "COMPLIANCE_TRAINING",
        "generation_prompt": "Act as a compliance trainer. Based *only* on the provided documents, generate 50 unique questions about required courses, due dates, and tracking.",
    },
    # 077
    {
        "category": "EXPORT_TRADE_COMPLIANCE",
        "generation_prompt": "Act as an export compliance analyst. Based *only* on the provided documents, generate 50 unique questions about screening, licensing, and documentation.",
    },
    # 078
    {
        "category": "DATA_PROTECTION_OFFICER",
        "generation_prompt": "Act as a DPO. Based *only* on the provided documents, generate 50 unique questions about DPIAs, lawful bases, and breach notifications.",
    },
    # 079
    {
        "category": "RECORDS_MANAGEMENT",
        "generation_prompt": "Act as a records manager. Based *only* on the provided documents, generate 50 unique questions about retention schedules, archival, and retrieval.",
    },
    # 080
    {
        "category": "KNOWLEDGE_MANAGEMENT",
        "generation_prompt": "Act as a knowledge manager. Based *only* on the provided documents, generate 50 unique questions about taxonomy, article templates, and review cycles.",
    },
    # 081
    {
        "category": "DOCUMENT_CONTROL",
        "generation_prompt": "Act as a document controller. Based *only* on the provided documents, generate 50 unique questions about versioning, approvals, and distribution lists.",
    },
    # 082
    {
        "category": "EXECUTIVE_ASSISTANT",
        "generation_prompt": "Act as an executive assistant. Based *only* on the provided documents, generate 50 unique questions about calendar norms, briefing docs, and travel rules.",
    },
    # 083
    {
        "category": "OFFICE_ADMIN",
        "generation_prompt": "Act as an office administrator. Based *only* on the provided documents, generate 50 unique questions about supplies, visitor policies, and mailroom steps.",
    },
    # 084
    {
        "category": "TRAVEL_COORDINATION",
        "generation_prompt": "Act as a travel coordinator. Based *only* on the provided documents, generate 50 unique questions about booking policy, per diem, and preferred vendors.",
    },
    # 085
    {
        "category": "REAL_ESTATE",
        "generation_prompt": "Act as a real estate manager. Based *only* on the provided documents, generate 50 unique questions about leases, moves, and site selection criteria.",
    },
    # 086
    {
        "category": "WORKPLACE_EXPERIENCE",
        "generation_prompt": "Act as a workplace experience lead. Based *only* on the provided documents, generate 50 unique questions about amenities, surveys, and events.",
    },
    # 087
    {
        "category": "RECEPTION_FRONTDESK",
        "generation_prompt": "Act as a receptionist. Based *only* on the provided documents, generate 50 unique questions about check-in, badges, and delivery handling.",
    },
    # 088
    {
        "category": "FIELD_SERVICE",
        "generation_prompt": "Act as a field service technician. Based *only* on the provided documents, generate 50 unique questions about dispatch, safety, and work orders.",
    },
    # 089
    {
        "category": "CUSTOMER_EDUCATION",
        "generation_prompt": "Act as a customer education manager. Based *only* on the provided documents, generate 50 unique questions about course catalogs, exams, and certifications.",
    },
    # 090
    {
        "category": "TRAINING_LD",
        "generation_prompt": "Act as an L&D coordinator. Based *only* on the provided documents, generate 50 unique questions about learning paths, competencies, and evaluations.",
    },
    # 091
    {
        "category": "TALENT_ACQUISITION",
        "generation_prompt": "Act as a recruiter. Based *only* on the provided documents, generate 50 unique questions about job postings, interview loops, and candidate comms.",
    },
    # 092
    {
        "category": "COMPENSATION_BENEFITS",
        "generation_prompt": "Act as a compensation and benefits analyst. Based *only* on the provided documents, generate 50 unique questions about pay bands, merit cycles, and enrollment.",
    },
    # 093
    {
        "category": "HRIS",
        "generation_prompt": "Act as an HRIS analyst. Based *only* on the provided documents, generate 50 unique questions about workflows, tickets, and data change approvals.",
    },
    # 094
    {
        "category": "PERFORMANCE_MANAGEMENT",
        "generation_prompt": "Act as a performance program lead. Based *only* on the provided documents, generate 50 unique questions about review cycles, calibrations, and goals.",
    },
    # 095
    {
        "category": "EMPLOYEE_RELATIONS",
        "generation_prompt": "Act as an employee relations partner. Based *only* on the provided documents, generate 50 unique questions about complaint intake, mediation, and policies.",
    },
    # 096
    {
        "category": "DIVERSITY_INCLUSION",
        "generation_prompt": "Act as a DEI lead. Based *only* on the provided documents, generate 50 unique questions about ERGs, training plans, and representation metrics.",
    },
    # 097
    {
        "category": "CSR_ESG",
        "generation_prompt": "Act as a CSR or ESG manager. Based *only* on the provided documents, generate 50 unique questions about reporting frameworks, initiatives, and volunteering.",
    },
    # 098
    {
        "category": "SUSTAINABILITY",
        "generation_prompt": "Act as a sustainability manager. Based *only* on the provided documents, generate 50 unique questions about emissions accounting, targets, and reductions.",
    },
    # 099
    {
        "category": "PUBLIC_AFFAIRS",
        "generation_prompt": "Act as a public affairs lead. Based *only* on the provided documents, generate 50 unique questions about lobbying policies and community engagement.",
    },
    # 100
    {
        "category": "INVESTOR_RELATIONS",
        "generation_prompt": "Act as an investor relations manager. Based *only* on the provided documents, generate 50 unique questions about earnings call logistics and blackout periods.",
    },
    # 101
    {
        "category": "BOARD_RELATIONS",
        "generation_prompt": "Act as a board liaison. Based *only* on the provided documents, generate 50 unique questions about meeting logistics, material deadlines, and agendas.",
    },
    # 102
    {
        "category": "CEO_OFFICE",
        "generation_prompt": "Act as a chief of staff. Based *only* on the provided documents, generate 50 unique questions about executive briefings, decision logs, and priorities.",
    },
    # 103
    {
        "category": "CFO_OFFICE",
        "generation_prompt": "Act as a CFO office analyst. Based *only* on the provided documents, generate 50 unique questions about finance calendars and policy references.",
    },
    # 104
    {
        "category": "CIO_OFFICE",
        "generation_prompt": "Act as a CIO office PM. Based *only* on the provided documents, generate 50 unique questions about IT strategy summaries and governance forums.",
    },
    # 105
    {
        "category": "CTO_OFFICE",
        "generation_prompt": "Act as a CTO office strategist. Based *only* on the provided documents, generate 50 unique questions about technology strategy and architecture standards.",
    },
    # 106
    {
        "category": "CISO_OFFICE",
        "generation_prompt": "Act as a CISO office analyst. Based *only* on the provided documents, generate 50 unique questions about program charters, KRIs, and reporting cadence.",
    },
    # 107
    {
        "category": "PRODUCT_MARKETING",
        "generation_prompt": "Act as a product marketing manager. Based *only* on the provided documents, generate 50 unique questions about messaging, positioning, and launch plans.",
    },
    # 108
    {
        "category": "PRICING",
        "generation_prompt": "Act as a pricing analyst. Based *only* on the provided documents, generate 50 unique questions about price change process, discount approvals, and guardrails.",
    },
    # 109
    {
        "category": "QUOTE_TO_CASH",
        "generation_prompt": "Act as a quote to cash lead. Based *only* on the provided documents, generate 50 unique questions about quoting flows, approvals, and billing triggers.",
    },
    # 110
    {
        "category": "ORDER_MANAGEMENT",
        "generation_prompt": "Act as an order management specialist. Based *only* on the provided documents, generate 50 unique questions about acceptance, fulfillment, and holds.",
    },
    # 111
    {
        "category": "BILLING",
        "generation_prompt": "Act as a billing specialist. Based *only* on the provided documents, generate 50 unique questions about invoice formats, billing cycles, and disputes.",
    },
    # 112
    {
        "category": "CUSTOMER_BILLING_SUPPORT",
        "generation_prompt": "Act as a billing support agent. Based *only* on the provided documents, generate 50 unique questions about dispute intake, credit memos, and adjustments.",
    },
    # 113
    {
        "category": "LEGAL_EDISCOVERY",
        "generation_prompt": "Act as an eDiscovery manager. Based *only* on the provided documents, generate 50 unique questions about hold notices, collection steps, and reviews.",
    },
    # 114
    {
        "category": "IT_ASSET_MANAGEMENT",
        "generation_prompt": "Act as an ITAM lead. Based *only* on the provided documents, generate 50 unique questions about lifecycle, tagging, stock levels, and disposal.",
    },
    # 115
    {
        "category": "ENDPOINT_ENGINEERING",
        "generation_prompt": "Act as an endpoint engineer. Based *only* on the provided documents, generate 50 unique questions about images, patch windows, and configuration baselines.",
    },
    # 116
    {
        "category": "NETWORK_ENGINEERING",
        "generation_prompt": "Act as a network engineer. Based *only* on the provided documents, generate 50 unique questions about change windows, standards, and monitoring.",
    },
    # 117
    {
        "category": "IDENTITY_ACCESS",
        "generation_prompt": "Act as an identity engineer. Based *only* on the provided documents, generate 50 unique questions about SSO patterns, MFA enrollment, and break-glass access.",
    },
    # 118
    {
        "category": "COLLAB_TOOLS",
        "generation_prompt": "Act as a collaboration tools admin. Based *only* on the provided documents, generate 50 unique questions about channels, meeting policies, and retention.",
    },
    # 119
    {
        "category": "SERVICE_DESK",
        "generation_prompt": "Act as a service desk lead. Based *only* on the provided documents, generate 50 unique questions about intake categories, priorities, and SLAs.",
    },
    # 120
    {
        "category": "CHANGE_MANAGEMENT",
        "generation_prompt": "Act as a change manager. Based *only* on the provided documents, generate 50 unique questions about CAB processes, change types, and approvals.",
    },
    # 121
    {
        "category": "RELEASE_MANAGEMENT",
        "generation_prompt": "Act as a release manager. Based *only* on the provided documents, generate 50 unique questions about calendars, readiness checks, and rollback steps.",
    },
    # 122
    {
        "category": "CONFIGURATION_MANAGEMENT",
        "generation_prompt": "Act as a CMDB admin. Based *only* on the provided documents, generate 50 unique questions about CI classes, relationships, and data quality.",
    },
    # 123
    {
        "category": "INCIDENT_MANAGEMENT",
        "generation_prompt": "Act as an incident manager. Based *only* on the provided documents, generate 50 unique questions about severity, bridges, and comms templates.",
    },
    # 124
    {
        "category": "PROBLEM_MANAGEMENT",
        "generation_prompt": "Act as a problem manager. Based *only* on the provided documents, generate 50 unique questions about RCA methods, KEDB entries, and follow-ups.",
    },
    # 125
    {
        "category": "CAPACITY_PLANNING",
        "generation_prompt": "Act as a capacity planner. Based *only* on the provided documents, generate 50 unique questions about forecasting methods, thresholds, and reviews.",
    },
    # 126
    {
        "category": "PERFORMANCE_ENGINEERING",
        "generation_prompt": "Act as a performance engineer. Based *only* on the provided documents, generate 50 unique questions about load tests, KPIs, and bottleneck triage.",
    },
    # 127
    {
        "category": "ARCHITECTURE",
        "generation_prompt": "Act as an enterprise architect. Based *only* on the provided documents, generate 50 unique questions about reference architectures and review boards.",
    },
    # 128
    {
        "category": "ENTERPRISE_APPS",
        "generation_prompt": "Act as an enterprise apps analyst. Based *only* on the provided documents, generate 50 unique questions about enhancement requests and ticket routing.",
    },
    # 129
    {
        "category": "ERP_SUPPORT",
        "generation_prompt": "Act as an ERP analyst. Based *only* on the provided documents, generate 50 unique questions about change requests, batch schedules, and roles.",
    },
    # 130
    {
        "category": "CRM_ADMIN",
        "generation_prompt": "Act as a CRM administrator. Based *only* on the provided documents, generate 50 unique questions about field governance, permissions, and integrations.",
    },
    # 131
    {
        "category": "HCM_ADMIN",
        "generation_prompt": "Act as an HCM admin. Based *only* on the provided documents, generate 50 unique questions about job changes, position management, and approvals.",
    },
    # 132
    {
        "category": "LMS_ADMIN",
        "generation_prompt": "Act as an LMS admin. Based *only* on the provided documents, generate 50 unique questions about course creation, enrollments, and reporting.",
    },
    # 133
    {
        "category": "PROCUREMENT_SOURCING",
        "generation_prompt": "Act as a sourcing manager. Based *only* on the provided documents, generate 50 unique questions about RFP steps, evaluation criteria, and timelines.",
    },
    # 134
    {
        "category": "VENDOR_MANAGEMENT",
        "generation_prompt": "Act as a vendor manager. Based *only* on the provided documents, generate 50 unique questions about scorecards, QBRs, and escalation paths.",
    },
    # 135
    {
        "category": "CONTRACT_LIFECYCLE_MGMT",
        "generation_prompt": "Act as a CLM admin. Based *only* on the provided documents, generate 50 unique questions about templates, fallback clauses, and approval routing.",
    },
    # 136
    {
        "category": "LICENSE_MANAGEMENT",
        "generation_prompt": "Act as a license manager. Based *only* on the provided documents, generate 50 unique questions about entitlements, audits, and true-ups.",
    },
    # 137
    {
        "category": "MOBILE_DEVICE_MGMT",
        "generation_prompt": "Act as an MDM admin. Based *only* on the provided documents, generate 50 unique questions about enrollment, compliance, and remote wipe.",
    },
    # 138
    {
        "category": "PRINT_SERVICES",
        "generation_prompt": "Act as a print services lead. Based *only* on the provided documents, generate 50 unique questions about quotas, secure print, and maintenance.",
    },
    # 139
    {
        "category": "TELECOM_UC",
        "generation_prompt": "Act as a telecom admin. Based *only* on the provided documents, generate 50 unique questions about number porting, call routing, and voicemail.",
    },
    # 140
    {
        "category": "REMOTE_WORK",
        "generation_prompt": "Act as a remote work coordinator. Based *only* on the provided documents, generate 50 unique questions about home office stipends, security, and etiquette.",
    },
    # 141
    {
        "category": "PHYSICAL_SECURITY",
        "generation_prompt": "Act as a physical security lead. Based *only* on the provided documents, generate 50 unique questions about badge policies, escorts, and camera retention.",
    },
    # 142
    {
        "category": "LAB_OPERATIONS",
        "generation_prompt": "Act as a lab operations manager. Based *only* on the provided documents, generate 50 unique questions about safety, equipment booking, and waste disposal.",
    },
    # 143
    {
        "category": "R_AND_D",
        "generation_prompt": "Act as an R&D scientist. Based *only* on the provided documents, generate 50 unique questions about experiment SOPs, documentation, and approvals.",
    },
    # 144
    {
        "category": "CLINICAL_TRIALS",
        "generation_prompt": "Act as a clinical operations lead. Based *only* on the provided documents, generate 50 unique questions about study phases, consent, and monitoring.",
    },
    # 145
    {
        "category": "REGULATORY_AFFAIRS",
        "generation_prompt": "Act as a regulatory affairs specialist. Based *only* on the provided documents, generate 50 unique questions about submissions, labeling changes, and audits.",
    },
    # 146
    {
        "category": "PRODUCT_SAFETY",
        "generation_prompt": "Act as a product safety manager. Based *only* on the provided documents, generate 50 unique questions about incident reporting and compliance testing.",
    },
    # 147
    {
        "category": "QUALITY_SYSTEMS",
        "generation_prompt": "Act as a quality systems owner. Based *only* on the provided documents, generate 50 unique questions about QMS modules, training, and audits.",
    },
    # 148
    {
        "category": "ENGINEERING_RELEASE",
        "generation_prompt": "Act as a release engineer. Based *only* on the provided documents, generate 50 unique questions about branching, release notes, and cutover plans.",
    },
    # 149
    {
        "category": "FEATURE_FLAGGING",
        "generation_prompt": "Act as a feature flag manager. Based *only* on the provided documents, generate 50 unique questions about rollout plans, kill switches, and targeting.",
    },
    # 150
    {
        "category": "A_B_TESTING",
        "generation_prompt": "Act as an experimentation lead. Based *only* on the provided documents, generate 50 unique questions about test design, guardrails, and analysis.",
    },
    # 151
    {
        "category": "CUSTOMER_FEEDBACK",
        "generation_prompt": "Act as a VOC manager. Based *only* on the provided documents, generate 50 unique questions about survey templates, NPS processes, and insights routing.",
    },
    # 152
    {
        "category": "COMMUNITY_MANAGEMENT",
        "generation_prompt": "Act as a community manager. Based *only* on the provided documents, generate 50 unique questions about moderation rules, forum guidelines, and responses.",
    },
    # 153
    {
        "category": "OPEN_SOURCE_COMPLIANCE",
        "generation_prompt": "Act as an open source program lead. Based *only* on the provided documents, generate 50 unique questions about license reviews and contributions.",
    },
    # 154
    {
        "category": "LOCALIZATION",
        "generation_prompt": "Act as a localization manager. Based *only* on the provided documents, generate 50 unique questions about translation workflows and in-context review.",
    },
    # 155
    {
        "category": "TRANSLATIONS",
        "generation_prompt": "Act as a translator. Based *only* on the provided documents, generate 50 unique questions about glossary usage, style preferences, and QA steps.",
    },
    # 156
    {
        "category": "CONTENT_STRATEGY",
        "generation_prompt": "Act as a content strategist. Based *only* on the provided documents, generate 50 unique questions about content pillars, governance, and audits.",
    },
    # 157
    {
        "category": "EDITORIAL",
        "generation_prompt": "Act as an editor. Based *only* on the provided documents, generate 50 unique questions about editorial style, fact checking, and approvals.",
    },
    # 158
    {
        "category": "COPYWRITING",
        "generation_prompt": "Act as a copywriter. Based *only* on the provided documents, generate 50 unique questions about tone of voice, briefs, and review cycles.",
    },
    # 159
    {
        "category": "VIDEO_PRODUCTION",
        "generation_prompt": "Act as a video producer. Based *only* on the provided documents, generate 50 unique questions about shot lists, equipment booking, and edits.",
    },
    # 160
    {
        "category": "PHOTO_MEDIA",
        "generation_prompt": "Act as a photographer. Based *only* on the provided documents, generate 50 unique questions about asset storage, releases, and usage rights.",
    },
    # 161
    {
        "category": "ECOMMERCE",
        "generation_prompt": "Act as an ecommerce manager. Based *only* on the provided documents, generate 50 unique questions about promotion rules, PDP standards, and checkout.",
    },
    # 162
    {
        "category": "RETAIL_OPERATIONS",
        "generation_prompt": "Act as a retail operations lead. Based *only* on the provided documents, generate 50 unique questions about store opening, cash handling, and audits.",
    },
    # 163
    {
        "category": "STORE_OPERATIONS",
        "generation_prompt": "Act as a store manager. Based *only* on the provided documents, generate 50 unique questions about staffing, inventory counts, and shrink prevention.",
    },
    # 164
    {
        "category": "MERCHANDISING",
        "generation_prompt": "Act as a merchandiser. Based *only* on the provided documents, generate 50 unique questions about planograms, seasonal calendars, and signage.",
    },
    # 165
    {
        "category": "SUPPLIER_RELATIONS",
        "generation_prompt": "Act as a supplier manager. Based *only* on the provided documents, generate 50 unique questions about onboarding, scorecards, and QBR agendas.",
    },
    # 166
    {
        "category": "DEMAND_PLANNING",
        "generation_prompt": "Act as a demand planner. Based *only* on the provided documents, generate 50 unique questions about forecast cadence, inputs, and overrides.",
    },
    # 167
    {
        "category": "INVENTORY_PLANNING",
        "generation_prompt": "Act as an inventory planner. Based *only* on the provided documents, generate 50 unique questions about safety stock, reorder rules, and counts.",
    },
    # 168
    {
        "category": "FLEET_MANAGEMENT",
        "generation_prompt": "Act as a fleet manager. Based *only* on the provided documents, generate 50 unique questions about maintenance schedules, fuel cards, and logs.",
    },
    # 169
    {
        "category": "SHIPPING",
        "generation_prompt": "Act as a shipping coordinator. Based *only* on the provided documents, generate 50 unique questions about packaging standards, labeling, and pickups.",
    },
    # 170
    {
        "category": "CUSTOMS_TRADE",
        "generation_prompt": "Act as a customs specialist. Based *only* on the provided documents, generate 50 unique questions about HS codes, documentation, and brokers.",
    },
    # 171
    {
        "category": "RETURNS_RMA",
        "generation_prompt": "Act as a returns coordinator. Based *only* on the provided documents, generate 50 unique questions about RMA policies, inspections, and refunds.",
    },
    # 172
    {
        "category": "WARRANTY_SUPPORT",
        "generation_prompt": "Act as a warranty agent. Based *only* on the provided documents, generate 50 unique questions about coverage terms, claim steps, and approvals.",
    },
    # 173
    {
        "category": "FIELD_SALES",
        "generation_prompt": "Act as a field sales rep. Based *only* on the provided documents, generate 50 unique questions about territory plans, meeting prep, and follow-ups.",
    },
    # 174
    {
        "category": "FRANCHISE_SUPPORT",
        "generation_prompt": "Act as a franchise operations lead. Based *only* on the provided documents, generate 50 unique questions about franchisee onboarding and support.",
    },
    # 175
    {
        "category": "HOSPITALITY_OPERATIONS",
        "generation_prompt": "Act as a hotel operations manager. Based *only* on the provided documents, generate 50 unique questions about guest service SOPs and checklists.",
    },
    # 176
    {
        "category": "FOOD_SAFETY",
        "generation_prompt": "Act as a food safety manager. Based *only* on the provided documents, generate 50 unique questions about HACCP procedures and temperature logs.",
    },
    # 177
    {
        "category": "SCIENCE_LAB_SAFETY",
        "generation_prompt": "Act as a lab safety officer. Based *only* on the provided documents, generate 50 unique questions about training, PPE, and incident reporting.",
    },
    # 178
    {
        "category": "EDUCATION_ADMIN",
        "generation_prompt": "Act as a school administrator. Based *only* on the provided documents, generate 50 unique questions about enrollment, attendance, and schedules.",
    },
    # 179
    {
        "category": "NONPROFIT_DEVELOPMENT",
        "generation_prompt": "Act as a nonprofit development manager. Based *only* on the provided documents, generate 50 unique questions about donor comms and campaigns.",
    },
    # 180
    {
        "category": "GRANTS_MANAGEMENT",
        "generation_prompt": "Act as a grants manager. Based *only* on the provided documents, generate 50 unique questions about application cycles, reviews, and reporting.",
    },
    # 181
    {
        "category": "GOVERNMENT_CONTRACTS",
        "generation_prompt": "Act as a government contracts manager. Based *only* on the provided documents, generate 50 unique questions about FAR clauses and deliverables.",
    },
    # 182
    {
        "category": "PUBLIC_SECTOR_COMPLIANCE",
        "generation_prompt": "Act as a public sector compliance lead. Based *only* on the provided documents, generate 50 unique questions about certifications and addenda.",
    },
    # 183
    {
        "category": "ENERGY_OPERATIONS",
        "generation_prompt": "Act as an energy operations manager. Based *only* on the provided documents, generate 50 unique questions about outages, maintenance, and safety.",
    },
    # 184
    {
        "category": "MINING_OPERATIONS",
        "generation_prompt": "Act as a mining operations lead. Based *only* on the provided documents, generate 50 unique questions about shift handovers and safety briefings.",
    },
    # 185
    {
        "category": "CONSTRUCTION_PROJECTS",
        "generation_prompt": "Act as a construction PM. Based *only* on the provided documents, generate 50 unique questions about RFIs, submittals, and change orders.",
    },
    # 186
    {
        "category": "REAL_ESTATE_DEVELOPMENT",
        "generation_prompt": "Act as a development manager. Based *only* on the provided documents, generate 50 unique questions about entitlements, milestones, and risks.",
    },
    # 187
    {
        "category": "INSURANCE_CLAIMS",
        "generation_prompt": "Act as a claims adjuster. Based *only* on the provided documents, generate 50 unique questions about claim intake, documentation, and adjudication.",
    },
    # 188
    {
        "category": "ACTUARIAL",
        "generation_prompt": "Act as an actuary. Based *only* on the provided documents, generate 50 unique questions about model documentation, assumptions, and governance.",
    },
    # 189
    {
        "category": "HEALTHCARE_OPERATIONS",
        "generation_prompt": "Act as a healthcare operations lead. Based *only* on the provided documents, generate 50 unique questions about patient flow, scheduling, and triage.",
    },
    # 190
    {
        "category": "PATIENT_SAFETY",
        "generation_prompt": "Act as a patient safety officer. Based *only* on the provided documents, generate 50 unique questions about incident reporting and checklists.",
    },
    # 191
    {
        "category": "MEDICAL_RECORDS",
        "generation_prompt": "Act as a health information manager. Based *only* on the provided documents, generate 50 unique questions about release of info and retention.",
    },
    # 192
    {
        "category": "TELEHEALTH_SUPPORT",
        "generation_prompt": "Act as a telehealth coordinator. Based *only* on the provided documents, generate 50 unique questions about platform usage, consent, and troubleshooting.",
    },
    # 193
    {
        "category": "GAMING_OPERATIONS",
        "generation_prompt": "Act as a game operations manager. Based *only* on the provided documents, generate 50 unique questions about live ops playbooks and safety rules.",
    },
    # 194
    {
        "category": "AD_TECH",
        "generation_prompt": "Act as an ad tech specialist. Based *only* on the provided documents, generate 50 unique questions about trafficking, privacy settings, and QA checks.",
    },
    # 195
    {
        "category": "MEDIA_PLANNING",
        "generation_prompt": "Act as a media planner. Based *only* on the provided documents, generate 50 unique questions about flighting, reach targets, and reporting.",
    },
    # 196
    {
        "category": "MARKET_RESEARCH",
        "generation_prompt": "Act as a research analyst. Based *only* on the provided documents, generate 50 unique questions about survey design, sampling, and incentives.",
    },
    # 197
    {
        "category": "COMPETITIVE_INTELLIGENCE",
        "generation_prompt": "Act as a competitive intelligence lead. Based *only* on the provided documents, generate 50 unique questions about profiles, win-loss, and monitoring.",
    },
    # 198
    {
        "category": "ETHICS_HOTLINE",
        "generation_prompt": "Act as an ethics coordinator. Based *only* on the provided documents, generate 50 unique questions about reporting channels, training, and protections.",
    },
    # 199
    {
        "category": "WHISTLEBLOWER_PROGRAM",
        "generation_prompt": "Act as a whistleblower program manager. Based *only* on the provided documents, generate 50 unique questions about intake, investigation steps, and outcomes.",
    },
    # 200
    {
        "category": "CHARITY_FOUNDATION",
        "generation_prompt": "Act as a corporate foundation manager. Based *only* on the provided documents, generate 50 unique questions about grantmaking policies and events.",
    },
]


JSON_FORMAT_INSTRUCTIONS = """
**Important:** You MUST return your answer as a single, valid JSON list of strings.
Do not include any preamble, introduction, or conversation (e.g., "Here are the prompts...").
Do not use markdown backticks (```json).
Your entire response must be *only* the JSON list.

Example Format:
["What is the company's policy on remote work?", "How do I submit an expense report?", "Where can I find the holiday calendar?"]
"""

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# helper functions

def is_file_empty(file_path):
    return not os.path.exists(file_path) or os.path.getsize(file_path) == 0

def write_header_if_needed(file_path):
    if is_file_empty(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "label", "category"])
        print(f"Created new file and wrote header to: {file_path}")

# MAIN EXECUTION

def main():
    print(f"Initializing Vertex AI for project {PROJECT_ID}...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Set up the Grounding Tool
    data_store_path = (
        f"projects/{PROJECT_ID}/locations/global/collections/"
        f"default_collection/dataStores/{DATA_STORE_ID}"
    )
    # Create VertexAISearch object
    vertex_ai_search = grounding.VertexAISearch(datastore=data_store_path)
    
    # Create retrieval object
    retrieval_tool = grounding.Retrieval(source=vertex_ai_search)
    
    # Create tool object from retrieval object
    grounding_tool = Tool.from_retrieval(retrieval_tool)
    
    # Initialize model
    model = GenerativeModel(
        MODEL_NAME,
        safety_settings=SAFETY_SETTINGS,
        tools=[grounding_tool]
    )
    
    print(f"Model {MODEL_NAME} initialized.")
    print(f"Grounding to data store: {DATA_STORE_ID}")

    write_header_if_needed(OUTPUT_CSV_FILE)

    # 4. Initialize progress and cost trackers
    total_prompts_generated = 0
    num_batches = len(META_PROMPTS)

    # generation loop
    with open(OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for i, meta_prompt in enumerate(META_PROMPTS):
            category = meta_prompt["category"]
            full_prompt_text = (
                f"{meta_prompt['generation_prompt']}\n\n{JSON_FORMAT_INSTRUCTIONS}"
            )
            
            print(f"\n--- Batch {i+1}/{num_batches} (Category: {category}) ---")
            
            response = None
            for attempt in range(MAX_RETRIES):
                try:
                    # Call API
                    print(f"  Attempt {attempt+1}/{MAX_RETRIES}: Sending prompt to Gemini...")
                    response = model.generate_content(
                        [full_prompt_text],
                    )
                    
                    # Parse JSON 
                    response_text = response.candidates[0].content.parts[0].text
                    generated_prompts = json.loads(response_text)
                    
                    if not isinstance(generated_prompts, list):
                        print(f"  ERROR: Model did not return a list. Retrying...")
                        time.sleep(5)
                        continue

                    # Write to CSV
                    prompts_in_batch = 0
                    for prompt_text in generated_prompts:
                        if isinstance(prompt_text, str) and prompt_text.strip():
                            writer.writerow([prompt_text.strip(), 0, category])
                            prompts_in_batch += 1
                    
                    total_prompts_generated += prompts_in_batch
                    
                    print(f"  Success: Wrote {prompts_in_batch} prompts.")
                    
                    break 

                except json.JSONDecodeError:
                    print(f"  ERROR: Failed to decode JSON from model response. Retrying...")
                    print(f"  Model output started with: {response_text[:200]}...")
                    time.sleep(5)
                except Exception as e:
                    print(f"  An unexpected error occurred: {e}. Retrying...")
                    time.sleep(5)
            
            # If all retries failed
            if response is None or not prompts_in_batch:
                print(f"  FATAL: Failed to process batch {i+1} after {MAX_RETRIES} attempts. Skipping.")
                

    print("\n-------------------------------------------------")
    print(f"Dataset generation complete.")
    print(f"Total new prompts added: {total_prompts_generated}")
    print(f"Data saved to: {OUTPUT_CSV_FILE}")
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()