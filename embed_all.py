"""
One-time script to embed all existing data into t_embeddings.
Run once after data migration. After this, new records get embedded automatically.
"""
import sys
sys.path.insert(0, '/home/claude/turkish')

# Set env vars for standalone run
import os
os.environ.setdefault('SUPABASE_URL', 'https://oprtrkmuoaxdinnxqosz.supabase.co')
os.environ.setdefault('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9wcnRya211b2F4ZGlubnhxb3N6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY3NTg0ODUsImV4cCI6MjA5MjMzNDQ4NX0.aloIaAH4d_wm65UuOVXQ_PGXbpZuTHgl08PWaGQZ5ps')
os.environ.setdefault('OPENAI_API_KEY', 'sk-proj-xsp_yHnUNOnyybtLTLphyG3thM6zLN17_3zA5Q67_7G09H2lligbdxF5HLVqmKKCIO_URHDsKCT3B1bkFJqz-LAS_Q1hZxdv52TVWTHEJvPywKEC5sIXeMky_YmMkeM65dwaptmMB_sEuV8MNjc6hnmG6-UA')
os.environ.setdefault('OPENAI_MODEL', 'gpt-4o')
os.environ.setdefault('EMBED_MODEL', 'text-embedding-3-small')

import agent, time, logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("embed_all")

def embed_table(table, rows, text_fn):
    count = 0
    for row in rows:
        try:
            text = text_fn(row)
            if text and len(text.strip()) > 5:
                agent.embed_and_store(table, row['id'], text)
                count += 1
                time.sleep(0.1)  # Rate limit safety
        except Exception as e:
            log.error("Error embedding %s id=%s: %s", table, row.get('id'), e)
    log.info("Embedded %d/%d from %s", count, len(rows), table)
    return count

total = 0

# Vacancies
rows = agent.sb_get("t_vacancies", limit=500)
total += embed_table("t_vacancies", rows, lambda r:
    f"Vacancy: {r.get('address','')}, {r.get('suburb','')}. "
    f"Size: {r.get('size_sqm','')} sqm. "
    f"Asking rent: ${r.get('asking_rent_pa','')} pa. "
    f"Status: {r.get('status','')}. "
    f"Vacating tenant: {r.get('vacating_tenant','')}. "
    f"Available: {r.get('available_date','')}."
)

# Requirements
rows = agent.sb_get("t_requirements", limit=500)
total += embed_table("t_requirements", rows, lambda r:
    f"Requirement: {r.get('company','')} is looking for industrial space. "
    f"Size: {r.get('size_min','')} to {r.get('size_max','')} sqm. "
    f"Location: {r.get('preferred_location','')}. "
    f"Budget: ${r.get('budget_pa','')} pa. "
    f"Timeline: {r.get('timeline','')}. "
    f"Status: {r.get('status','')}. "
    f"Priority: {r.get('priority','')}."
)

# Deals
rows = agent.sb_get("t_deals", limit=500)
total += embed_table("t_deals", rows, lambda r:
    f"Deal: {r.get('tenant','')} leased from {r.get('landlord','')} "
    f"at {r.get('address','')}. "
    f"Size: {r.get('size_sqm','')} sqm. "
    f"Rent: ${r.get('rent_pa','')} pa. "
    f"Term: {r.get('term_years','')} years. "
    f"Status: {r.get('status','')}. "
    f"Commission: ${r.get('commission','')}."
)

# Properties
rows = agent.sb_get("t_properties", limit=500)
total += embed_table("t_properties", rows, lambda r:
    f"Property: {r.get('address','')}, {r.get('suburb','')}. "
    f"Type: {r.get('property_type','')}. "
    f"Size: {r.get('size_sqm','')} sqm. "
    f"Status: {r.get('status','')}. "
    f"Landlord: {r.get('landlord','')}. "
    f"Occupier: {r.get('occupier','')}. "
    f"Lease expiry: {r.get('lease_expiry','')}."
)

# Style profile
rows = agent.sb_get("t_style_profile", limit=100)
total += embed_table("t_style_profile", rows, lambda r:
    f"About Michael — {r.get('category','')}/{r.get('key','')}: {r.get('value','')}"
)

log.info("DONE. Total embedded: %d records", total)
print(f"\n✅ Embedded {total} records into pgvector")
