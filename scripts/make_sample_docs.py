from pathlib import Path

from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "data" / "sample"

DOCS = {
    "pump_station_manual.pdf": [
        ("Inspection schedule",
         "The main hydraulic pump must be inspected every 200 operating hours. The inspection covers "
         "the housing, the bearings and the drive coupling. Record the hour meter reading in the "
         "logbook after each inspection. If the pump runs continuously, the inspection falls due "
         "roughly every eight days."),
        ("Seal replacement",
         "Mechanical seals are replaced once a year, or sooner if leakage exceeds ten drops per "
         "minute. Shut down and isolate the pump before opening the seal chamber. Use only seals "
         "from the approved parts list, and note the batch number in the maintenance record."),
        ("Emergency shutdown",
         "In an emergency press the red stop button on the control cabinet. The pump stops within "
         "three seconds and the inlet valve closes automatically. After a shutdown, the shift "
         "supervisor must be informed before the pump is restarted. A restart requires a second "
         "person to confirm the pipe pressure is below two bar."),
        ("Lubrication",
         "Bearings are lubricated every 500 operating hours with lithium grease of grade NLGI 2. "
         "Apply two pumps of the grease gun per bearing. Over-greasing raises the bearing "
         "temperature and should be avoided."),
        ("Vibration limits",
         "Vibration is measured monthly at the bearing housing. A reading above 7.1 millimetres per "
         "second means the pump has to be taken out of service. Readings between 4.5 and 7.1 are "
         "marked as a warning and measured again after one week."),
        ("Spare parts",
         "A spare impeller and two sets of seals are kept in the storage cabinet in room B2. The key "
         "is held by the shift supervisor. When a spare is used, it has to be reordered within five "
         "working days."),
    ],
    "employee_handbook.pdf": [
        ("Vacation",
         "Full-time employees receive thirty vacation days per year. Up to five unused days may be "
         "carried over to the next year and must be taken before the end of March. Vacation "
         "requests are submitted through the HR portal at least two weeks in advance."),
        ("Remote work",
         "Employees may work remotely up to two days per week after the end of the probation period. "
         "Remote days are agreed with the team lead and entered in the shared calendar. Core hours, "
         "during which everyone is reachable, are from ten to three."),
        ("Sick leave",
         "If you are unable to work because of illness, inform your team lead before nine o'clock on "
         "the first day. A medical certificate is required from the fourth calendar day of absence. "
         "Sick days are not deducted from vacation."),
        ("Expense reports",
         "Business expenses are reimbursed when the receipt is submitted within thirty days. Travel "
         "by train is booked in second class. Meals during business trips are reimbursed up to "
         "twenty-eight euros per day."),
        ("Onboarding",
         "New employees receive a laptop and an access card on their first day. A buddy from the "
         "team guides them through the first four weeks. The probation period lasts six months, "
         "with a review meeting after three months."),
        ("Equipment return",
         "When leaving the company, return the laptop, the access card and any other equipment on "
         "the last working day. IT wipes returned laptops within five working days. The HR team "
         "confirms the return in writing."),
    ],
    "it_operations_guide.pdf": [
        ("Backups",
         "Backups of all production databases run nightly at 02:00. Backups are kept for ninety days "
         "and stored in a second data centre. A restore test is carried out on the first Monday of "
         "every month."),
        ("Password policy",
         "Passwords must have at least fourteen characters. Multi-factor authentication is mandatory "
         "for all accounts with access to production systems. Passwords are changed only after a "
         "suspected compromise, not on a fixed schedule."),
        ("Incident response",
         "Security incidents are reported to the on-call engineer within thirty minutes of "
         "discovery. The on-call engineer opens an incident ticket and assigns a severity from one "
         "to four. Severity one incidents require a written report within two working days."),
        ("Patching",
         "Operating system patches are installed during the maintenance window on the second Tuesday "
         "of each month between 22:00 and 24:00. Critical security patches may be installed outside "
         "the window after approval by the head of IT."),
        ("Access requests",
         "Access to systems is requested through the service desk and approved by the line manager. "
         "Approved requests are processed within two working days. Access rights are reviewed every "
         "six months and unused accounts are disabled after ninety days of inactivity."),
        ("Monitoring alerts",
         "The monitoring system sends an alert when disk usage exceeds eighty-five percent or when "
         "response time stays above two seconds for five minutes. Alerts go to the on-call engineer "
         "by phone call. If the alert is not acknowledged within ten minutes, it is escalated to "
         "the team lead."),
    ],
}


def write_pdf(path: Path, pages: list[tuple[str, str]]) -> None:
    # invariant=1 keeps the output identical between runs
    pdf = canvas.Canvas(str(path), invariant=1)
    for heading, body in pages:
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(72, 770, heading)
        pdf.setFont("Helvetica", 11)
        y = 740
        for line in simpleSplit(body, "Helvetica", 11, 450):
            pdf.drawString(72, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCS.items():
        write_pdf(OUT / name, pages)
        print(f"wrote {name} ({len(pages)} pages)")
