from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

"""Сделать роутер чтобы с тг опрвить case_number и сформировать отчет пдф"""

def generate_pdf(case_data, filename):
    """
    Генерирует PDF-документ с данными о деле.
    """
    # Создаем документ
    pdf = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Заголовок
    title = Paragraph(f"Детали дела: {case_data.get('CaseNumber', 'не указано')}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Основная информация
    info = [
        ["ID дела", case_data.get("CaseId", "не указано")],
        ["Тип дела", case_data.get("CaseType", "не указано")],
        ["Дата начала", case_data.get("StartDate", "не указано")],
        ["Статус", case_data.get("State", "не указано")],
        ["Завершено", "Да" if case_data.get("Finished") else "Нет"]
    ]
    info_table = Table(info, colWidths=[200, 200])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 12))

    # Истцы
    plaintiffs = case_data.get("Plaintiffs", [])
    if plaintiffs:
        story.append(Paragraph("Истцы:", styles['Heading2']))
        plaintiff_data = [["Имя", "Адрес", "ИНН", "ОГРН"]] + [
            [p.get("Name", ""), p.get("Address", ""), p.get("Inn", ""), p.get("Ogrn", "")]
            for p in plaintiffs
        ]
        plaintiff_table = Table(plaintiff_data, colWidths=[150, 150, 100, 100])
        plaintiff_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(plaintiff_table)
        story.append(Spacer(1, 12))

    # Ответчики
    respondents = case_data.get("Respondents", [])
    if respondents:
        story.append(Paragraph("Ответчики:", styles['Heading2']))
        respondent_data = [["Имя", "Адрес", "ИНН", "ОГРН"]] + [
            [r.get("Name", ""), r.get("Address", ""), r.get("Inn", ""), r.get("Ogrn", "")]
            for r in respondents
        ]
        respondent_table = Table(respondent_data, colWidths=[150, 150, 100, 100])
        respondent_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(respondent_table)
        story.append(Spacer(1, 12))

    # Документы
    documents = []
    for instance in case_data.get("CaseInstances", []):
        if "File" in instance and instance["File"]:
            documents.append({
                "Тип": "Решение по инстанции",
                "Название": instance["File"].get("Name", "не указано"),
                "Ссылка": instance["File"].get("URL", "не указано")
            })
        for event in instance.get("InstanceEvents", []):
            if "File" in event and event["File"]:
                documents.append({
                    "Тип": event.get("EventTypeName", "не указано"),
                    "Название": event.get("DocumentName", "не указано"),
                    "Ссылка": event["File"]
                })
    
    if documents:
        story.append(Paragraph("Документы:", styles['Heading2']))
        doc_data = [["Тип", "Название", "Ссылка"]] + [
            [d["Тип"], d["Название"], d["Ссылка"]] for d in documents
        ]
        doc_table = Table(doc_data, colWidths=[100, 200, 200])
        doc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(doc_table)
        story.append(Spacer(1, 12))

    # Собираем PDF
    pdf.build(story)