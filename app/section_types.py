"""THE section_type contract — the single definition the labels, the validator, the eval schema and shrew-server's
SECTION_ENUM all derive from (hub decision 2026-09-14: the enum grows to match the labels; labels are not mapped
down — except typos, near-duplicates and singletons, which are folded here and were rewritten in train v2.6.4).

    from shrew_ocr.section_types import SECTION_TYPES, canonical

History: v2 shipped an 8-value enum (metrics_v2.SECTION_ENUM). The broadsheet (2026-08) and dense-harvest (2026-09)
labels were written with an open vocabulary and reached 466 distinct values on train v2.6.3, 400+ of them with
fewer than 40 chunks (teacher-invented one-offs like "diplomatic_diplomacy", "service_station", "导读"). shrew-server
rejected everything outside the 8 and coerced 13% of pages. This list is the server agent's baseline (original 8 + broadsheet 9 + dense 7 + other)
reconciled against the counts: baseline values with real mass kept as-is, near-duplicates folded, and 12 region kinds
with hundreds of chunks each added; every observed legacy value maps onto one of them via
LEGACY_MAP / canonical().
"""
from __future__ import annotations

import re

# (value, one-line definition). Order = documentation order; nothing depends on it.
SECTION_TYPE_DEFS = [
    # --- document structure (v2 lineage; the original 8) ---
    ("abstract", "abstract / executive summary printed as such"),
    ("introduction", "introduction, background, motivation, preface, related work"),
    ("methodology", "methods, materials, procedures, experimental setup"),
    ("results", "results, findings, measurements"),
    ("discussion", "discussion, analysis of results, limitations"),
    ("conclusion", "conclusion, recommendations, future work"),
    ("appendix", "appendix, references, supplementary material, footnotes"),
    ("technical_content", "any other prose section of a technical/legal/business document"),
    # --- generic page text (dense lineage) ---
    ("body", "generic running text with no document structure (dense scans, dictionary entries)"),
    ("table_text", "prose that belongs to a table region but is not the table body (table notes)"),
    ("list", "a list of items printed as a list (not a data grid)"),
    # --- newspaper / magazine regions (broadsheet + dense lineage) ---
    ("news_article", "a news story: hard news, reports, dispatches, sports/business/science news"),
    ("news_brief", "short news items, briefs, briefings, digest entries"),
    ("news_analysis", "news analysis, investigative or special report, explanatory analysis"),
    ("feature_article", "feature story, profile, interview, human-interest, travel, biography, literary pieces"),
    ("opinion", "opinion piece, column, essay, letters to the editor, reviews"),
    ("commentary", "commentary / 评论 pieces labelled as such (kept distinct from opinion at the labels' own usage)"),
    ("editorial", "the paper's own editorial"),
    ("preview", "front-page refers, teasers and previews pointing to inside pages or other sections"),
    ("preview_list", "a list of previews / refers (NYT front page)"),
    ("headline", "a standalone headline, sub-headline, kicker or banner not attached to its body chunk"),
    ("header", "running page header, section header, page title line"),
    ("masthead", "nameplate, masthead, dateline block, edition/price line"),
    ("title", "a title line / title block on a dense page (magazine cover, manual title page)"),
    ("footer", "page footer, folio, colophon, imprint, credits, bylines-only blocks, page furniture"),
    ("index", "index, table of contents, section index, navigation, page guide, link lists"),
    ("photo_caption", "photo, figure, chart or artwork captions and credits"),
    ("photo_feature", "photo features, photo essays, galleries, picture news"),
    ("stat_box", "statistics box, data chart, infographic, scoreboard, timeline, data table rendered as text"),
    ("sidebar", "sidebar, info box, quote box, summary box, guide, tips, Q&A, service information, editor's notes"),
    ("advertisement", "advertisements, advertorials, classifieds, promos, listings"),
    ("legal_notice", "public/legal/government notices and announcements"),
    ("official_document", "reprinted laws, regulations, policies, speeches, statements, official documents"),
    ("obituary", "obituaries and death notices"),
    ("weather_box", "weather boxes, forecasts and reports"),
    # --- fallback ---
    ("other", "text that fits no other value; unknown/missing normalises here (logged, never fails a page)"),
]
SECTION_TYPES: list[str] = [v for v, _ in SECTION_TYPE_DEFS]
SECTION_SET = frozenset(SECTION_TYPES)

# Explicit folds for legacy values that the rule engine below would get wrong or that deserve a stated choice.
_EXPLICIT = {
    "body": "body", "article": "news_article", "document_content": "technical_content", "academic_paper": "technical_content",
    "academic": "technical_content", "academic_article": "technical_content", "academic_debate": "opinion", "theory": "technical_content",
    "theoretical_analysis": "news_analysis", "research_report": "news_analysis", "report": "news_article", "reportage": "news_article",
    "meeting_report": "news_article", "policy_news": "news_article", "news_dispatch": "news_article", "lead_article": "news_article",
    "sub_article": "news_article", "article_section": "news_article", "article_continuation": "news_article", "continuation": "news_article",
    "story": "feature_article", "personal_story": "feature_article", "personal_narrative": "feature_article", "personal_essay": "opinion", "essay": "opinion",
    "literary_essay": "feature_article", "column": "opinion",  "forum": "opinion", "forum_article": "opinion",
    "criticism": "opinion", "critics_notebook": "opinion", "design_notebook": "opinion", "art_review": "opinion", "book_review": "opinion",
    "editorial_note": "sidebar", "editorial_info": "footer", "editorial_staff": "footer", 
    "editorial_commentary": "editorial", "kicker": "headline", "kicker_headline": "headline", "banner": "headline", "banner_headline": "headline",
     "title_page": "masthead", "title_block": "masthead", "heading": "header", "subtitle": "headline",
    "cover": "masthead", "cover_line": "preview", "cover_teaser": "preview", "cover_flash": "preview", "front_matter": "masthead",
    "front_page": "preview", "front_page_header": "masthead", "front_page_masthead": "masthead", "front_page_index": "index",
    "front_page_links": "index", "front_page_previews": "preview", "front_page_teaser": "preview", "front_page_briefs": "news_brief",
    "dateline": "masthead", "folio": "footer", "identifier": "footer", "page_info": "footer", "header_info": "masthead", "masthead_info": "masthead",
    "metadata": "footer", "meta": "footer", "meta_info": "footer", "label": "footer", "ui_element": "footer", "page_furniture": "footer",
    "stamp": "footer", "logo": "footer", "motto": "masthead", "credits": "footer", "caption_credits": "photo_caption", "byline": "footer",
    "contributor_list": "footer", "imprint": "footer", "colophon": "footer", "note": "footer", "notes": "sidebar", "annotation": "footer",
    "marginalia": "footer", "info": "sidebar", "box": "sidebar", "box_text": "sidebar", "standing_box": "sidebar", "standalone_box": "sidebar",
    "feature_box": "sidebar", "quote": "sidebar", "pull_quote": "sidebar", "policy_quote": "sidebar", "summary": "sidebar",
    "summary_headline": "headline", "guide": "sidebar", "service": "sidebar", "service_window": "sidebar", "service_station": "sidebar",
    "service_column": "sidebar", "service_article": "sidebar", "service_guide": "sidebar", "tips": "sidebar", "suggestion": "sidebar",
    "feedback": "sidebar", "extended_reading": "sidebar", "supplementary": "sidebar", "supplementary_reading": "sidebar", "utility": "sidebar",
    "qa": "sidebar", "qa_box": "sidebar", "question": "sidebar", "policy_qa": "sidebar", "consumer_guide": "sidebar", "travel_guide": "sidebar",
    "health_tips": "sidebar", "travel_tips": "sidebar", "table": "stat_box", "figure": "other", "image": "other", "photo": "other",
    "map": "other", "diagram": "stat_box", "graphic": "stat_box", "graphic_element": "stat_box", "graphic_explainer": "stat_box",
    "feature_graphic": "stat_box", "artwork": "other", "artwork_caption": "photo_caption", "puzzle": "other", "cartoon": "other",
    "comic": "other", "cartoon_commentary": "opinion", "comic_commentary": "opinion", "timeline": "stat_box", "chart": "stat_box",
    "chart_box": "stat_box", "data_box": "stat_box", "data_list": "stat_box", "statistical_data": "stat_box", "polling_report": "stat_box",
    "interactive": "other", "interactive_element": "other", "interactive_feature": "other", "gallery": "photo_feature",
    "image_gallery": "photo_feature", "photo_gallery": "photo_feature", "photo_essay": "photo_feature", "picture_news": "photo_feature",
    "image_news": "photo_feature", "image_report": "photo_feature", "photo_report": "photo_feature", "photo_reportage": "photo_feature",
    "photo_news": "photo_feature", "photo_feature": "photo_feature", "photo_story": "photo_feature", "classified": "advertisement",
    "directory": "advertisement", "auction": "advertisement", "magazine_listing": "advertisement", "promo": "advertisement",
    "media_promo": "advertisement", "feature_promo": "advertisement", "call_for_submissions": "legal_notice", "call_for_letters": "legal_notice",
    "announcement": "legal_notice", "government_announcement": "legal_notice", "notice": "legal_notice", "government notice": "legal_notice",
    "public notice": "legal_notice", "government_notice": "legal_notice", "policy_notice": "legal_notice", "boxed_notice": "legal_notice",
    "standing_notice": "legal_notice", "ad_notice": "advertisement", "prospectus": "official_document", "declaration": "official_document",
    "speech": "official_document", "statement": "official_document", "statement_box": "official_document", "political_statement": "official_document",
    "official_statement": "official_document", "official_report": "official_document", "document": "official_document",
    "document_transcript": "official_document", "document_exhibit": "official_document", "historical_document": "official_document",
    "government_document": "official_document", "whitepaper_content": "official_document", "whitepaper_toc": "index", "proposal": "official_document",
    "proposal_summary": "official_document", "law": "official_document", "law_text": "official_document", "law_section": "official_document",
    "law_header": "official_document", "law_regulation": "official_document", "regulation": "official_document", "legal": "legal_notice",
    "legal_case": "news_article", "legal_document": "official_document", "legal_explanation": "news_analysis", "policy": "official_document",
    "policy_document": "official_document", "policy_interpretation": "news_analysis", "policy_report": "news_analysis", "policy_analysis": "news_analysis",
    "data_analysis": "news_analysis", "data_report": "news_analysis", "statistical_report": "news_analysis", "science_report": "news_article",
    "investigative_report": "news_analysis", "investigation": "news_analysis", "special_report": "news_analysis", "opinion_analysis": "opinion",
    "expert_view": "opinion", "expert_commentary": "opinion", "short_commentary": "opinion", "political_commentary": "opinion",
    "series_commentary": "opinion", "column_commentary": "opinion", "column_opinion": "opinion", "column_header": "header",
    "reader_letter": "opinion", "letter_to_editor": "opinion", "opinion_letter": "opinion", "letter": "opinion",
    "poem": "feature_article", "poem_box": "feature_article", "poetry": "feature_article", "novel": "feature_article", "literature": "feature_article", "excerpt": "feature_article",
    "diary": "feature_article", "travelogue": "feature_article", "memoir": "feature_article", "biography": "feature_article", "biography_box": "sidebar",
    "profile": "feature_article", "person_profile": "feature_article", "company_profile": "feature_article", "interview": "feature_article", "interview_section": "feature_article",
    "case_study": "feature_article", "history_article": "feature_article", "historical_feature": "feature_article", "historical_review": "feature_article",
    "historical_narrative": "feature_article", "history_feature": "feature_article", "cultural_history": "feature_article", "cultural_historical": "feature_article",
    "cultural_feature": "feature_article", "encyclopedia_entry": "body", "entry": "body", "reference": "appendix",
    "intro": "introduction", "intro_text": "introduction", "preface": "introduction", "feature_intro": "introduction",
    "headline_intro": "headline", "toc": "index", "contents": "index", "navigation": "index", "navigation_box": "index", "links": "index",
    "related_links": "index", "link_box": "index", "refer": "preview", "page_guide": "index", "guide_index": "index", "导读": "index",
    "misc": "other", "miscellaneous": "other", "": "other", "community": "other", "lifestyle": "feature_article", "food": "feature_article", "arts": "opinion",
    "sports": "news_article", "business": "news_article", "science": "news_article", "magazine": "preview", "special_section": "preview",
    "brief": "news_brief", "briefs": "news_brief", "briefing": "news_brief", "briefing_box": "news_brief", "briefs_box": "news_brief",
    "brief_news": "news_brief", "list_news": "news_brief", "news_archive": "news_brief", "advertising": "advertisement", "ad": "advertisement",
    "advertorial": "advertisement", "ad_feature": "advertisement", "ad_article": "advertisement", "advertising_feature": "advertisement",
    "academic_ad": "advertisement", "listicle": "list", "book_list": "list", "sidebar_quote": "sidebar", "sidebar_news": "news_brief",
    "infographic": "stat_box", "info_graphic": "stat_box", "infographic_explainer": "stat_box", "data_chart": "stat_box",
    "data_visualization": "stat_box", "stat-box": "stat_box", "standalone_stat_box": "stat_box", "weather": "weather_box",
    "image_caption": "photo_caption", "image_captions": "photo_caption",  "caption_text": "photo_caption", "figure_caption": "photo_caption",
    "graphic_caption": "photo_caption", "obituary_note": "obituary", "obituary_index": "obituary", "obituary_teaser": "preview", "obituary_preview": "preview",
    "feature_story": "feature_article", "feature_article": "feature_article", "feature_header": "headline", "feature_headline": "headline", "feature_teaser": "preview",
    "feature_preview": "preview", "news_analysis": "news_analysis", "analysis_headline": "headline", "news_headline": "headline", "news_headlines": "headline",
    "headlines": "headline", "headline_list": "headline", "headlines_list": "headline", "headline_block": "headline", "headline_box": "headline",
    "headline_display": "headline", "headline_image": "other", "article_headline": "headline", "sub_headline": "headline", "subheadline": "headline",
    "subhead": "headline", "subheading": "headline", "sports_headline": "headline", "section_headline": "headline", "section_header": "header",
    "section_index": "index", "index_list": "index", "index_section": "index", "index_box": "index", "table_of_contents": "index",
    "toc_section": "index", "magazine_toc": "index", "magazine_index": "index", "magazine_contents": "index", "magazine_table_of_contents": "index",
    "opinion_article": "opinion", "opinion_piece": "opinion", "opinion_column": "opinion", "opinion_essay": "opinion", "opinion_brief": "news_brief",
    "opinion_preview": "preview", "opinion_teaser": "preview", "news_brief": "news_brief", "news_briefs": "news_brief", "news_briefing": "news_brief",
    "news_preview": "preview", "news_teaser": "preview",  "preview_box": "preview", "preview_section": "preview",
    "preview_headline": "preview", "preview_block": "preview", "preview_link": "preview", "preview_links": "preview", "preview_boxes": "preview",
    "previews": "preview", "page_previews": "preview", "section_preview": "preview", "section_teaser": "preview", "teaser": "preview",
    "teasers": "preview", "teaser_block": "preview", "teaser_group": "preview", "teaser_list": "preview", "sports_preview": "preview",
    "arts_preview": "preview", "business_preview": "preview", "food_preview": "preview", "science_preview": "preview", "styles_preview": "preview",
    "magazine_preview": "preview", "metropolitan_preview": "preview", "arts_teaser": "preview", "business_teaser": "preview", "food_teaser": "preview",
    "sports_teaser": "preview", "sports_brief": "news_brief", "sports_briefs": "news_brief", "arts_brief": "news_brief", "arts_briefs": "news_brief",
    "business_brief": "news_brief", "science_brief": "news_brief", "styles_brief": "news_brief", "sports_summary": "news_brief",
    "summary_box": "sidebar", "summary_list": "sidebar", "info_box": "sidebar", "guide_box": "sidebar", "tip_box": "sidebar", "quote_box": "sidebar",
    "sidebar_box": "sidebar", "sidebar_info": "sidebar",  "weather_report": "weather_box", "weather_forecast": "weather_box",
    "sports_article": "news_article", "sports_news": "news_article", "sports_feature": "feature_article", "business_news": "news_article",
    "business_feature": "feature_article", "financial_news": "news_article", "health_news": "news_article", "health_article": "news_article",
    "health_column": "opinion", "science_article": "news_article", "science_feature": "feature_article", "arts_feature": "feature_article",
    "travel_feature": "feature_article", "policy_document": "official_document", "official_document": "official_document", "government notice": "legal_notice",
}

# Rule engine for anything not listed explicitly (future teacher inventions resolve deterministically).
_RULES = [
    (r"(^|_)(preview|teaser|refer)s?($|_)", "preview"), (r"(^|_)briefs?($|_)|briefing", "news_brief"),
    (r"headline|kicker|subhead|banner", "headline"), (r"caption", "photo_caption"), (r"masthead|nameplate|dateline|cover", "masthead"),
    (r"(^|_)(toc|index|contents|navigation|links?)($|_)", "index"), (r"advert|(^|_)ads?($|_)|classified|promo|listing", "advertisement"),
    (r"notice|announcement", "legal_notice"), (r"law|regulation|policy|official|statement|declaration|document|speech|whitepaper", "official_document"),
    (r"obituar", "obituary"), (r"weather", "weather_box"), (r"stat|chart|graphic|infograph|data|table|diagram|timeline", "stat_box"),
    (r"photo|gallery|picture|image", "photo_feature"), (r"illustration|cartoon|comic|artwork|drawing|map", "other"),
    (r"sidebar|box|guide|tips?|service|qa|question|summary|quote|supplement", "sidebar"),
    (r"letter", "opinion"), (r"review|critic", "opinion"), (r"poem|poetry|novel|literar|fiction|excerpt|diary", "feature_article"),
    (r"editorial", "editorial"), (r"opinion|commentary|column|essay|forum|debate", "opinion"),
    (r"analysis|investigat|report", "news_analysis"), (r"feature|profile|interview|biograph|memoir|travel|history|cultural|story", "feature_article"),
    (r"footer|colophon|imprint|credit|byline|folio|meta|furniture", "footer"), (r"header|title|heading", "header"),
    (r"news|article|dispatch|sports|business|science|health|financ", "news_article"),
    (r"abstract", "abstract"), (r"intro|preface|background|motivation|related", "introduction"), (r"method|material|procedure", "methodology"),
    (r"result|finding", "results"), (r"discussion", "discussion"), (r"conclusion|recommend|future", "conclusion"), (r"appendix|reference|bibliograph", "appendix"),
    (r"footnote|endnote", "appendix"), (r"(^|_)list($|_)", "list"), (r"body|entry|text|paragraph|content", "body"),
]


def canonical(value) -> str:
    """Map any section_type string to a canonical SECTION_TYPES value. Idempotent on canonical values."""
    if value is None:
        return "other"
    v = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if v in SECTION_SET:
        return v
    if v in _EXPLICIT:
        return _EXPLICIT[v]
    if str(value) in _EXPLICIT:
        return _EXPLICIT[str(value)]
    for rx, target in _RULES:
        if re.search(rx, v):
            return target
    return "other"


LEGACY_MAP = dict(_EXPLICIT)  # exported for ledgers/tools; canonical() is the function to call


if __name__ == "__main__":
    import json, sys
    counts = json.load(open(sys.argv[1]))["counts"] if len(sys.argv) > 1 else {}
    moved = {}
    for v, n in counts.items():
        c = canonical(v)
        if c != v:
            moved.setdefault(c, []).append((v, n))
    print(f"{len(SECTION_TYPES)} canonical values; {len(counts)} observed; {sum(len(x) for x in moved.values())} folded")
    for c, items in sorted(moved.items(), key=lambda x: -sum(n for _, n in x[1])):
        print(f"  -> {c:18} {sum(n for _, n in items):7d} chunks from {len(items)} values: " + ", ".join(f"{v}({n})" for v, n in sorted(items, key=lambda x: -x[1])[:8]) + (" …" if len(items) > 8 else ""))
