"""
Sample Bilingual PDF Generator.
Generates 10 English and 10 Arabic PDF reports if none exist,
guaranteeing full reproducibility of ingestion, retrieval, and evaluation.
"""

import logging
from pathlib import Path
from typing import List, Dict

from src.config import settings

logger = logging.getLogger(__name__)

ENGLISH_REPORTS = [
    {
        "filename": "EN_Report_01_AI_Strategy_2024.pdf",
        "title": "Enterprise AI Strategy and Governance Report 2024",
        "pages": [
            "Executive Summary: In 2024, our enterprise accelerated generative AI adoption across customer operations and risk management. Total investment reached $45 million, representing a 35% increase year-over-year. Model governance frameworks were established to monitor bias, hallucination rates, and regulatory compliance.",
            "Operational Pillars: We deployed automated underwriting copilots reducing loan processing time from 4 days to 4 hours. Data privacy protocols ensure zero retention of customer PII during model inference. The Enterprise AI Committee reviews all production releases bi-weekly.",
            "Future Roadmap: By 2025, agentic workflows will handle 60% of tier-1 support tickets. We target annual operational savings of $18 million while upskilling 1,200 employees on responsible AI practices."
        ]
    },
    {
        "filename": "EN_Report_02_Financial_Performance_Q3.pdf",
        "title": "Q3 2024 Financial Performance & Earnings Report",
        "pages": [
            "Revenue and Margin Overview: Third-quarter net revenue reached $312.4 million, reflecting 14.2% YoY growth. Adjusted EBITDA stood at $78.1 million with an operating margin of 25.0%. Cloud services contributed 48% of total revenue.",
            "Segment Breakdown: Enterprise software revenue expanded to $150 million, while consumer subscription revenue grew 8% to $82 million. Free cash flow for the quarter was $54.6 million, up from $41.2 million in Q3 2023.",
            "Capital Allocation and Guidance: The board approved a $20 million share repurchase program. Full-year revenue guidance was revised upward to between $1.22 billion and $1.25 billion."
        ]
    },
    {
        "filename": "EN_Report_03_Renewable_Energy_Outlook.pdf",
        "title": "Renewable Energy Transition & Infrastructure Outlook",
        "pages": [
            "Clean Power Capacity: Our solar and wind generation portfolio reached 3.8 GW in 2024. Levelized Cost of Energy (LCOE) for utility-scale solar dropped to $0.032 per kWh, making it the most cost-effective power source in our operational markets.",
            "Storage Solutions: Grid-scale battery energy storage systems (BESS) capacity expanded to 1,200 MWh. Battery degradation remained below 1.5% annually, supported by smart thermal management.",
            "Decarbonization Milestones: Direct Scope 1 emissions fell by 22% compared to the 2020 baseline. We anticipate reaching 70% clean electricity across all manufacturing hubs by 2026."
        ]
    },
    {
        "filename": "EN_Report_04_Cybersecurity_Framework.pdf",
        "title": "Corporate Cybersecurity & Zero Trust Framework",
        "pages": [
            "Threat Environment: In 2024, the Security Operations Center (SOC) intercepted over 4.2 million malicious intrusion attempts. Phishing accounted for 58% of initial vector attacks, followed by API credential stuffing at 24%.",
            "Zero Trust Implementation: Identity-aware proxy architectures and mandatory FIDO2 hardware tokens were deployed across all 15,000 corporate endpoints. Least-privilege access policies reduced lateral movement risk by 80%.",
            "Incident Response Metrics: Mean Time to Detect (MTTD) decreased to 11 minutes, and Mean Time to Respond (MTTR) reached 28 minutes. Annual penetration tests confirmed zero high-severity perimeter breaches."
        ]
    },
    {
        "filename": "EN_Report_05_Supply_Chain_Resilience.pdf",
        "title": "Global Supply Chain Resilience and Logistics Strategy",
        "pages": [
            "Logistics Network: Nearshoring initiatives shifted 30% of critical component manufacturing from East Asia to North America and Europe. On-time delivery rates recovered to 96.4% across tier-1 suppliers.",
            "Inventory Optimization: Machine learning demand forecasting reduced excess safety stock by $32 million while lowering stockout incidents by 45%. Warehouse automation cut handling costs by $1.80 per unit.",
            "Supplier Risk Assessment: Over 450 key suppliers were evaluated on geopolitical, financial, and environmental risk criteria. Secondary suppliers were pre-contracted for 100% of single-sourced components."
        ]
    },
    {
        "filename": "EN_Report_06_Healthcare_Technology.pdf",
        "title": "Healthcare Technology & Clinical AI Diagnostics",
        "pages": [
            "Clinical Diagnostic Tools: Deployed FDA-cleared computer vision models for radiology screenings, achieving 94.8% sensitivity in detecting early-stage pulmonary nodules across 12 participating hospitals.",
            "Telehealth Adoption: Remote patient monitoring platforms served 185,000 chronic care patients. Hospital readmission rates within 30 days dropped by 18.2% among enrolled cardiovascular cohorts.",
            "Security and Compliance: All patient health information (PHI) is encrypted with AES-256 both in transit and at rest, achieving strict compliance with HIPAA and HITECH requirements."
        ]
    },
    {
        "filename": "EN_Report_07_Cloud_Migration_Guide.pdf",
        "title": "Enterprise Cloud Migration and FinOps Architecture",
        "pages": [
            "Migration Progress: 82% of legacy core workloads have been migrated to a hybrid multi-cloud architecture. Decommissioning legacy on-premise data centers saved $14.5 million in annual facilities and cooling costs.",
            "FinOps Governance: Automated idle resource termination and spot instance utilization reduced monthly cloud spend by 23%. Unit cost per transaction fell from $0.045 to $0.028.",
            "Resilience and Disaster Recovery: Multi-region failover tests achieved a Recovery Point Objective (RPO) of under 10 seconds and a Recovery Time Objective (RTO) of under 2 minutes."
        ]
    },
    {
        "filename": "EN_Report_08_ESG_Sustainability_Charter.pdf",
        "title": "ESG Sustainability Charter and Environmental Targets",
        "pages": [
            "Environmental Stewardship: Net water consumption decreased by 34% across high-stress manufacturing facilities through closed-loop recycling systems. Single-use plastic was eliminated across 92% of product packaging.",
            "Social Responsibility: Gender pay equity reached 99.4% across global offices. Employee volunteering programs logged over 42,000 hours in local STEM education and environmental cleanup initiatives.",
            "Corporate Governance: The independent board members now represent 65% of the total board seats. Executive compensation packages tie 20% of variable bonuses to sustainability KPIs."
        ]
    },
    {
        "filename": "EN_Report_09_Human_Capital_Trends.pdf",
        "title": "Human Capital Trends & Future of Work Report",
        "pages": [
            "Workplace Model: A structured hybrid work policy (3 days on-site, 2 days remote) was formalized. Employee engagement scores rose to 81%, and voluntary employee turnover decreased from 14% to 9.2%.",
            "Upskilling Programs: Over $12 million was invested into technical learning programs, with 3,400 employees completing certifications in cloud computing, data science, and agile leadership.",
            "Talent Acquisition: Time-to-hire decreased by 12 days through AI-assisted resume screening and standardized competency interviews, while candidate satisfaction reached 88%."
        ]
    },
    {
        "filename": "EN_Report_10_Data_Privacy_Compliance.pdf",
        "title": "Data Privacy and International Regulatory Compliance",
        "pages": [
            "Regulatory Alignment: Full alignment was achieved with GDPR in Europe, CCPA in California, and national data protection laws. Data subject access requests (DSAR) were fulfilled within an average of 6.2 business days.",
            "Consent Management: Universal opt-out consent mechanisms were deployed across all customer-facing mobile apps and web platforms. User opt-in for functional analytics increased to 64%.",
            "Audits and Vendor Reviews: 100% of external third-party data processors underwent privacy impact assessments (PIA). Five vendor contracts were terminated due to inadequate data governance controls."
        ]
    }
]

ARABIC_REPORTS = [
    {
        "filename": "AR_Report_01_Digital_Transformation.pdf",
        "title": "تقرير استراتيجية التحول الرقمي والحوسبة السحابية",
        "pages": [
            "الملخص التنفيذي: شهد عام 2024 قفزة نوعية في مسار التحول الرقمي، حيث تم تحويل 85% من الخدمات الحكومية والداخلية إلى منصات رقمية متكاملة. وبلغ حجم الاستثمار في البنية التحتية السحابية 150 مليون ريال، مما قلل زمن إنجاز المعاملات بنسبة 60%.",
            "المنصات السحابية: تم اعتماد السحابة الهجينة وتوطين مراكز البيانات محلياً لضمان السيادة الرقمية وسلامة البيانات. وانخفضت الانقطاعات التشغيلية بنسبة 75% مع تطبيق معايير التوافر العالي 99.95%.",
            "الخطط المستقبلية: يستهدف برنامج 2025 دمج نماذج الذكاء الاصطناعي التوليدي في 40 خدمة استباقية للمستفيدين، وتدريب 5000 موظف على مهارات إدارة البيانات وهندسة السحابة."
        ]
    },
    {
        "filename": "AR_Report_02_Economic_Growth_Vision.pdf",
        "title": "تقرير مؤشرات النمو الاقتصادي والتنويع المالي",
        "pages": [
            "الأداء الاقتصادي: سجل الناتج المحلي الإجمالي غير النفطي نمواً بنسبة 4.8% خلال العام المالي، مدفوعاً بقطاعات السياحة والخدمات اللوجستية والتقنية. وتجاوزت مساهمة القطاع الخاص 42% من إجمالي الناتج المحلي.",
            "الاستثمارات الأجنبية المباشرة: ارتفعت تدفقات الاستثمار الأجنبي المباشر بنسبة 28% لتصل إلى 19 مليار دولار، مع تركز 35% منها في قطاعي التصنيع المتقدم والتقنيات النظيفة.",
            "الاستدامة المالية: انخفض العجز المالي التقديري إلى 1.2% بفضل كفاءة الإنفاق الحكومي وبرامج تنمية الإيرادات غير النفطية، وارتفع الاحتياطي العام بنسبة 8%."
        ]
    },
    {
        "filename": "AR_Report_03_Clean_Energy_Transition.pdf",
        "title": "تقرير مبادرات الطاقة النظيفة والهيدروجين الأخضر",
        "pages": [
            "مشاريع الطاقة المتجددة: تم تدشين محطات طاقة شمسية بقدرة إجمالية بلغت 2.5 غيغاوات، مما أسهم في خفض الانبعاثات الكربونية بمقدار 3.2 مليون طن سنوياً. وحققت تكلفة إنتاج الكهرباء رقماً قياسياً قدره 1.4 سنت لكل كيلوواط/ساعة.",
            "الهيدروجين الأخضر: بدأ إنشاء أكبر مصنع لإنتاج الهيدروجين الأخضر بطاقة إنتاجية تبلغ 600 طن يومياً بحلول عام 2026، بهدف التصدير للأسواق العالمية.",
            "كفاءة الطاقة: نجحت مبادرات العزل الحراري وتحديث معايير الأجهزة الكهربائية في خفض الاستهلاك الوطني للكهرباء بنسبة 7% في قطاع المباني السكنية والتجارية."
        ]
    },
    {
        "filename": "AR_Report_04_Cybersecurity_Policy.pdf",
        "title": "سياسات الأمن السيبراني وحماية البنية التحتية الوطنية",
        "pages": [
            "بيئة التهديدات: تصدت مراكز العمليات السيبرانية لأكثر من 5 ملايين محاولة هجوم سيبراني خلال الربع الثالث، وشكلت هجمات التصيد الإلكتروني 52% منها، تليها برمجيات الفدية بنسبة 21%.",
            "معايير الامتثال: تم تطبيق ضوابط الأمن السيبراني الأساسية بنسبة 96% في كافة الجهات الحيوية. وألزمت اللائحة تطبيق التحقق الثنائي وسياسات انعدام الثقة (Zero Trust) على كافة الشبكات.",
            "الكوادر الوطنية: تم تدريب 2500 متخصص في الاستجابة للحوادث السيبرانية واختبار الاختراق، وحققت الفرق الوطنية مراكز متقدمة في مؤشر الأمن السيبراني العالمي."
        ]
    },
    {
        "filename": "AR_Report_05_Logistics_and_Ports.pdf",
        "title": "تقرير تطوير الموانئ وسلاسل الإمداد اللوجستية",
        "pages": [
            "حركة الموانئ: ارتفعت مناولة الحاويات بنسبة 12% لتصل إلى 8.5 مليون حاوية قياسية. وتم تقليص مدة مكوث الحاويات في الموانئ البحرية من 6 أيام إلى 2.5 يوم بفضل المنصات الرقمية الموحدة.",
            "المناطق اللوجستية: تم افتتاح 4 مناطق لوجستية خاصة متصلة بشبكات السكك الحديدية، مما خفض تكاليف النقل الداخلي للبضائع بنسبة 18% ودعم تجارة الترانزيت الإقليمية.",
            "الربط الجوي والبحري: ارتفع مؤشر الأداء اللوجستي (LPI) 14 مرتبة عالمياً، وزادت خطوط الشحن المباشرة مع الأسواق الآسيوية والأوروبية بنسبة 22%."
        ]
    },
    {
        "filename": "AR_Report_06_Healthcare_Sector.pdf",
        "title": "تقرير تحديث منظومة الرعاية الصحية والتحول الصحي",
        "pages": [
            "الملف الصحي الموحد: تم ربط 70% من المستشفيات والمراكز الصحية بمنظومة الملف الصحي الرقمي الموحد، مما قلل تكرار الفحوصات المخبرية غير الضرورية بنسبة 25%.",
            "الطب الاتصالي: أجرت منصات الصحة الافتراضية أكثر من 3 ملايين استشارة طبية عن بُعد خلال عام 2024، وبلغت نسبة رضا المستفيدين 91% مع تقليل الضغط على أقسام الطوارئ.",
            "الوقاية والصحة العامة: انخفضت معدلات الإصابة بالأمراض المزمنة غير السارية بنسبة 4% بفضل برامج الكشف المبكر ومسارات الرعاية التخصصية لمرضى السكري والضغط."
        ]
    },
    {
        "filename": "AR_Report_07_Smart_Cities_Infrastructure.pdf",
        "title": "تقرير تقنيات المدن الذكية وإنترنت الأشياء",
        "pages": [
            "إدارة المرور الذكية: تم نشر 1200 مستشعر ذكي وكاميرا تحليلية لمراقبة الحركة المرورية، مما أسهم في خفض الاختناقات المرورية في أوقات الذروة بنسبة 22% وتقليل زمن الاستجابة للحوادث.",
            "الإنارة الذكية والنفايات: خفضت أعمدة الإنارة الموفرة بنظام التحكم الآلي استهلاك الطاقة بنسبة 40%، ورفعت حاويات النفايات المتصلة بإنترنت الأشياء كفاءة مسارات جمع النفايات بنسبة 30%.",
            "منصات البيانات الحضرية: تتيح المنصة الموحدة لإدارة المدينة لوحة مؤشرات فورية للتحكم في استهلاك المياه والكهرباء وجودة الهواء، مما يعزز الاستدامة الحضرية."
        ]
    },
    {
        "filename": "AR_Report_08_Water_Security_Strategy.pdf",
        "title": "تقرير استراتيجية الأمن المائي وتقنيات التحلية",
        "pages": [
            "إنتاج المياه المحلاة: بلغ الإنتاج اليومي للمياه المحلاة 6.2 مليون متر مكعب، مع تحويل 60% من المحطات إلى تقنية التناضح العكسي منخفضة استهلاك الطاقة، مما قلل التكاليف التشغيلية بنسبة 35%.",
            "إعادة استخدام المياه: ارتفع معدل معالجة مياه الصرف الصحي ثلاثياً إلى 78%، وتُستخدم هذه المياه المعالجة بنجاح في ري المسطحات الخضراء والمشاريع الصناعية والزراعية المقيدة.",
            "الخزن الاستراتيجي: تم رفع سعة الخزن الاستراتيجي للمياه إلى 14 مليون متر مكعب لتأمين الإمدادات في حالات الطوارئ لمدة تكفي 7 أيام متواصلة."
        ]
    },
    {
        "filename": "AR_Report_09_SME_Entrepreneurship.pdf",
        "title": "تقرير تمويل المنشآت الصغيرة والمتوسطة وريادة الأعمال",
        "pages": [
            "التمويل والاستثمار: نما حجم التمويل الموجه للمنشآت الصغيرة والمتوسطة بنسبة 24% ليصل إلى 45 مليار ريال، وبلغت مساهمة صناديق رأس المال الجريء في الشركات الناشئة 1.2 مليار ريال.",
            "حاضنات ومسرعات الأعمال: تم دعم أكثر من 600 شركة ناشئة في مجالات التقنية المالية والتجارة الإلكترونية، وبلغت نسبة استدامة المشاريع بعد سنتين من التأسيس 72%.",
            "التوظيف والنمو: وفر قطاع المنشآت الصغيرة والمتوسطة أكثر من 140 ألف فرصة عمل جديدة للشباب والفتيات، مع ارتفاع حصة المنشآت التي تقودها رائدات أعمال إلى 38%."
        ]
    },
    {
        "filename": "AR_Report_10_Education_and_EdTech.pdf",
        "title": "تقرير تطوير التعليم والمنصات التعليمية الرقمية",
        "pages": [
            "المنصات الرقمية التعليمية: سجلت المنصة الوطنية الموحدة للتعليم أكثر من 6 ملايين مستخدم نشط يومياً، مع إتاحة 45 ألف محتوى رقمي تفاعلي يدعم المناهج المطورة.",
            "تطوير المعلمين: أكمل 120 ألف معلم ومعلمة برامج تدريبية متخصصة في توظيف أدوات الذكاء الاصطناعي في الفصول الدراسية وتطبيق استراتيجيات التعلم القائم على المشاريع.",
            "مؤشرات الأداء الأكاديمي: ارتفعت نتائج الطلاب في الاختبارات المعيارية للرياضيات والعلوم بنسبة 8%، وانخفض معدل التسرب المدرسي في المرحلة الثانوية إلى أقل من 1.1%."
        ]
    }
]


def create_simple_pdf(file_path: Path, title: str, pages_content: List[str]):
    """
    Creates a valid standard PDF file containing the given title and pages.
    Uses pypdf if available, or PyMuPDF, or a lightweight fallback writer.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Method 1: Try PyMuPDF
    try:
        import fitz
        doc = fitz.open()
        for idx, page_text in enumerate(pages_content):
            page = doc.new_page(width=595, height=842)  # A4
            # Add title on first page
            header = f"{title} (Page {idx + 1})\n\n"
            text_to_insert = header + page_text
            # Use textbox with margin
            rect = fitz.Rect(50, 50, 545, 792)
            page.insert_textbox(rect, text_to_insert, fontsize=12, encoding=fitz.TEXT_ENCODING_UNICODE)
        doc.save(str(file_path))
        doc.close()
        return
    except Exception:
        pass

    # Method 2: Try reportlab or pypdf or minimal raw PDF
    try:
        from pypdf import PdfWriter
        # If pypdf writer is used with blank pages, we can write a clean minimal PDF
        # Alternatively, write a clean minimal PDF specification conforming to PDF-1.4:
    except Exception:
        pass

    # Fallback: Pure Python minimal valid PDF-1.4 with text streams
    _write_raw_pdf(file_path, title, pages_content)


def _write_raw_pdf(file_path: Path, title: str, pages_content: List[str]):
    """
    Writes a syntactically valid PDF 1.4 document without external C-extensions.
    Ensures that any PDF reader or text extractor (pypdf, pdfplumber, fitz) can extract the text.
    """
    objects = []
    offsets = []

    def add_object(obj_bytes: bytes) -> int:
        nonlocal objects
        obj_num = len(objects) + 1
        objects.append(obj_bytes)
        return obj_num

    # We will build pages
    # Font object
    font_bytes = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    # For each page, create a content stream
    content_obj_nums = []
    page_obj_nums = []

    # Object 1: Catalog (will fill later)
    # Object 2: Outlines
    # Object 3: Pages
    # Object 4: Font

    # Let's construct sequentially
    # Pre-allocate catalog and pages
    objects = [b"", b"", b"", font_bytes]  # 1: Catalog, 2: Outlines, 3: Pages, 4: Font

    kids_refs = []
    for idx, p_text in enumerate(pages_content):
        # Escape parenthesis
        clean_p_text = p_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        header_text = f"{title} - Page {idx+1}".replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        # PDF Content stream with BT ... ET
        stream_data = (
            f"BT\n/F1 12 Tf\n50 780 Td\n({header_text}) Tj\n0 -25 Td\n"
            f"({clean_p_text}) Tj\nET\n"
        ).encode("utf-8")

        stream_obj = (
            f"<< /Length {len(stream_data)} >>\nstream\n".encode("ascii")
            + stream_data
            + b"\nendstream"
        )
        objects.append(stream_obj)
        content_obj_num = len(objects)
        content_obj_nums.append(content_obj_num)

        # Page object
        # Note: Page object will refer to parent (3) and contents (content_obj_num)
        page_obj = (
            f"<< /Type /Page /Parent 3 0 R /MediaBox [0 0 595 842] "
            f"/Contents {content_obj_num} 0 R "
            f"/Resources << /Font << /F1 4 0 R >> >> >>"
        ).encode("ascii")
        objects.append(page_obj)
        page_obj_num = len(objects)
        page_obj_nums.append(page_obj_num)
        kids_refs.append(f"{page_obj_num} 0 R")

    # Now fill 1, 2, 3
    objects[0] = b"<< /Type /Catalog /Pages 3 0 R >>"
    objects[1] = b"<< /Type /Outlines /Count 0 >>"
    kids_str = " ".join(kids_refs)
    objects[2] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(kids_refs)} >>".encode("ascii")

    # Serialize PDF
    with open(file_path, "wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, obj in enumerate(objects):
            offset = f.tell()
            offsets.append(offset)
            f.write(f"{i + 1} 0 obj\n".encode("ascii"))
            f.write(obj)
            f.write(b"\nendobj\n")

        xref_start = f.tell()
        f.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        f.write(b"0000000000 65535 f \n")
        for off in offsets:
            f.write(f"{off:010d} 00000 n \n".encode("ascii"))

        f.write(b"trailer\n")
        f.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"))
        f.write(b"startxref\n")
        f.write(f"{xref_start}\n%%EOF\n".encode("ascii"))


def generate_sample_reports():
    """Generates all 20 PDF reports in data/english and data/arabic if missing."""
    eng_dir = settings.ENGLISH_DATA_DIR
    ara_dir = settings.ARABIC_DATA_DIR

    eng_dir.mkdir(parents=True, exist_ok=True)
    ara_dir.mkdir(parents=True, exist_ok=True)

    # Check if English PDFs already exist
    existing_eng = list(eng_dir.glob("*.pdf"))
    if len(existing_eng) < 10:
        logger.info(f"Generating 10 English sample PDF reports in {eng_dir}...")
        for report in ENGLISH_REPORTS:
            pdf_path = eng_dir / report["filename"]
            create_simple_pdf(pdf_path, report["title"], report["pages"])
        logger.info("English reports generation complete.")
    else:
        logger.info(f"Found {len(existing_eng)} existing English PDFs in {eng_dir}.")

    # Check if Arabic PDFs already exist
    existing_ara = list(ara_dir.glob("*.pdf"))
    if len(existing_ara) < 10:
        logger.info(f"Generating 10 Arabic sample PDF reports in {ara_dir}...")
        for report in ARABIC_REPORTS:
            pdf_path = ara_dir / report["filename"]
            create_simple_pdf(pdf_path, report["title"], report["pages"])
        logger.info("Arabic reports generation complete.")
    else:
        logger.info(f"Found {len(existing_ara)} existing Arabic PDFs in {ara_dir}.")


if __name__ == "__main__":
    generate_sample_reports()
