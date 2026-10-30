"""Generate PDF and DOCX knowledge-base files (run once during Phase 1)."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "knowledge_base"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyPearlz", parent=styles["BodyText"], fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="HPearlz", parent=styles["Heading1"], fontSize=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2Pearlz", parent=styles["Heading2"], fontSize=12, spaceAfter=6))
    return styles


def write_product_manual() -> None:
    path = KB / "product_manual_aquapearl500.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, title="AquaPearl 500 Pro Product Manual")
    styles = _styles()
    story: list = []

    story.append(Paragraph("AquaPearl 500 Pro — Product Manual", styles["HPearlz"]))
    story.append(Paragraph("Pearlz Home Systems (Pvt.) Ltd. · Version 2026.1 · Fictional training document", styles["BodyPearlz"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("1. Box contents", styles["H2Pearlz"]))
    for item in [
        "AquaPearl 500 Pro main unit",
        "Sediment pre-filter cartridge (installed)",
        "Carbon block filter (installed)",
        "RO membrane (installed)",
        "Remineralisation cartridge (installed)",
        "Inlet valve adapter kit and tubing",
        "Power adapter (220–240 V AC)",
        "Quick-start card and warranty card",
    ]:
        story.append(Paragraph(f"• {item}", styles["BodyPearlz"]))

    story.append(Paragraph("2. Technical specifications", styles["H2Pearlz"]))
    spec = [
        ["Parameter", "Value"],
        ["Flow rate (purified)", "Up to 1.8 L/min"],
        ["Tank capacity", "6.0 L"],
        ["Dimensions (W×D×H)", "38 × 22 × 42 cm"],
        ["Power", "220–240 V, 50 Hz, 60 W max"],
        ["Filter stages", "Sediment → Carbon → RO → Remineralisation (+ UV)"],
        ["Inlet pressure range", "0.1–0.4 MPa"],
    ]
    t = Table(spec, colWidths=[7 * cm, 8 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F33")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F4EF")),
            ]
        )
    )
    story.append(t)
    story.append(PageBreak())

    story.append(Paragraph("3. Control panel and indicator lights", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "Power LED (white): solid = on. Wi-Fi LED (blue): blinking = pairing, solid = connected. "
            "Filter LED (amber): on when a filter is due. Fault LED (red): blinks with audible alert codes.",
            styles["BodyPearlz"],
        )
    )

    story.append(Paragraph("4. Operating instructions", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "Open the inlet valve, plug in the adapter, and press Power. Allow the tank to fill before first draw. "
            "Use only potable municipal or bore water within the pressure range in §2.",
            styles["BodyPearlz"],
        )
    )

    story.append(Paragraph("5. Filter replacement intervals", styles["H2Pearlz"]))
    intervals = [
        ["Filter", "Typical interval"],
        ["Sediment pre-filter", "3–6 months"],
        ["Carbon block", "6–12 months"],
        ["RO membrane", "24 months"],
        ["Remineralisation", "12 months"],
        ["UV lamp (service)", "As indicated / ~9000 hours"],
    ]
    t2 = Table(intervals, colWidths=[7 * cm, 8 * cm])
    t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    story.append(t2)

    story.append(Paragraph("6. Cleaning and descaling schedule", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "Wipe exterior monthly with a damp cloth. Descale the tank every 6 months using Pearlz descaling solution "
            "only. Do not use abrasive cleaners on the control panel.",
            styles["BodyPearlz"],
        )
    )

    story.append(Paragraph("7. Safety warnings", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "Do not immerse the unit. Disconnect power before filter changes. Do not open the UV bay. "
            "This appliance improves aesthetic water quality per its design; it is not a medical device and "
            "does not diagnose or treat disease. Follow local public-health guidance for drinking water safety.",
            styles["BodyPearlz"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("8. Audible alert codes (E-01 to E-08)", styles["H2Pearlz"]))
    codes = [
        ["Code", "Pattern", "Meaning", "First action"],
        ["E-01", "1 long beep repeating", "No inlet water", "Open inlet valve"],
        ["E-02", "2 beeps repeating", "Tank full sensor fault", "Power cycle; clean contacts"],
        ["E-03", "3 short beeps then stop", "Sediment pre-filter saturation", "Replace pre-filter; hold RESET 5s"],
        ["E-04", "4 beeps", "Carbon filter due (fw ≥2.4)", "Replace carbon; reset"],
        ["E-05", "Blinking red LED", "Leak sensor wet", "Power off; dry sensor"],
        ["E-06", "5 beeps", "Pump overcurrent", "Power off; contact support"],
        ["E-07", "6 beeps", "UV lamp fault", "Do not open UV bay; contact support"],
        ["E-08", "7 beeps", "System fault", "Power off; contact support"],
    ]
    t3 = Table(codes, colWidths=[1.8 * cm, 3.5 * cm, 5 * cm, 5 * cm])
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D9488")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t3)

    story.append(Paragraph("9. Filter maintenance (page reference section)", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "To replace the sediment pre-filter: close inlet, depressurise, twist the sediment housing "
            "counter-clockwise, insert a new Pearlz AP500-SED cartridge, reseal, open inlet, and flush "
            "per the Installation Guide. After E-03, press and hold RESET for five seconds to clear the alert.",
            styles["BodyPearlz"],
        )
    )

    doc.build(story)
    print(f"Wrote {path}")


def write_installation_guide() -> None:
    path = KB / "installation_guide.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, title="Pearlz Installation Guide")
    styles = _styles()
    story: list = []

    story.append(Paragraph("Pearlz Installation Guide — AquaPearl Series", styles["HPearlz"]))
    story.append(Paragraph("Version 2026.1", styles["BodyPearlz"]))

    story.append(Paragraph("1. Pre-installation checklist", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "Confirm water pressure 0.1–0.4 MPa, a grounded 3-pin socket, and at least 10 cm clearance on each side "
            "for ventilation. Do not install outdoors or in direct sun.",
            styles["BodyPearlz"],
        )
    )

    story.append(Paragraph("2. Tools required", styles["H2Pearlz"]))
    story.append(Paragraph("Adjustable spanner, PTFE tape, bucket, clean cloth, spirit level.", styles["BodyPearlz"]))

    story.append(Paragraph("3. Wall-mount installation (summary)", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "Mark bracket holes, drill suitable anchors, hang the unit level, connect inlet tubing with PTFE tape, "
            "connect drain where required, and plug in only after leak-check with water on and power still off.",
            styles["BodyPearlz"],
        )
    )

    story.append(Paragraph("4. Under-sink installation (summary)", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "Shut municipal supply, install the saddle or adapter valve on the cold line, route tubing without kinks, "
            "secure the unit upright, and restore supply slowly while watching for drips.",
            styles["BodyPearlz"],
        )
    )

    story.append(Paragraph("5. First-run flushing procedure (required)", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "After installation or after changing carbon/RO filters, flush before drinking: "
            "(1) Fill and discard the first full tank. "
            "(2) Fill again and discard a second full tank. "
            "(3) On AquaPearl 500 Pro, run continuous dispense for 5 minutes on the third fill. "
            "This removes carbon fines and preservative taste. Skipping flush is the most common cause of "
            "“weird taste after I changed the filter.”",
            styles["BodyPearlz"],
        )
    )

    story.append(Paragraph("6. Professional installation", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "Book SKU SVC-INSTALL. Standard professional installation includes bracket mounting or under-sink hookup "
            "within 3 metres of the cold inlet and a basic leak check. Extra plumbing beyond that is quoted separately. "
            "List price is in the Pricing Guide.",
            styles["BodyPearlz"],
        )
    )

    story.append(Paragraph("7. What voids warranty during self-installation", styles["H2Pearlz"]))
    story.append(
        Paragraph(
            "Cracked fittings from overtightening, use of incompatible tubing, bypassing the leak sensor, or powering "
            "the unit while submerged connections are leaking may void warranty coverage for the resulting damage. "
            "See Warranty Policy exclusions.",
            styles["BodyPearlz"],
        )
    )

    doc.build(story)
    print(f"Wrote {path}")


def write_service_agreement() -> None:
    path = KB / "service_maintenance_agreement.docx"
    document = Document()
    document.add_heading("Pearlz Service & Maintenance Agreement", level=1)
    document.add_paragraph("Version 2026.1 · Last updated 2026-01-15")

    document.add_heading("1. Annual maintenance contract tiers", level=2)
    document.add_paragraph(
        "AMC Basic (SKU SVC-AMC-BASIC): one preventive visit per year, filter inspection, basic descaling labour. "
        "Parts billed separately unless under warranty."
    )
    document.add_paragraph(
        "AMC Plus (SKU SVC-AMC-PLUS): two preventive visits per year, priority scheduling, labour for covered wear parts "
        "listed in the Plus schedule; filters still billed as consumables."
    )

    document.add_heading("2. Visit frequency and response-time SLAs", level=2)
    document.add_paragraph("AMC Basic response target for breakdown visits: within 3 working days in metro areas.")
    document.add_paragraph("AMC Plus response target: within 1 working day in metro areas; 3 working days remote.")

    document.add_heading("3. Parts covered vs billed", level=2)
    document.add_paragraph(
        "Warranty-period defects in body, pump, and electronics follow the Warranty Policy. "
        "Consumable filters are always billed. Out-of-warranty labour is billed at published rates."
    )

    document.add_heading("4. Cancellation and pro-rata refund", level=2)
    document.add_paragraph(
        "AMC may be cancelled within 14 days of purchase for a full refund if no visit occurred. "
        "After a visit, refunds are pro-rata minus the visit cost."
    )

    document.add_heading("5. Warranty claim procedure (authoritative checklist)", level=2)
    document.add_paragraph(
        "This section is the claim procedure referenced by the Warranty Policy. Follow all steps:"
    )
    for step in [
        "Email support@pearlzhome.example with subject WARRANTY CLAIM and the serial number.",
        "Attach a clear photo of the invoice showing purchase date and model.",
        "Attach a photo of the serial label on the unit.",
        "Describe the fault, including any alert code (E-01–E-08).",
        "For leak-related claims, attach a short video of the leak location with power disconnected.",
        "Support issues a Claim ID within 2 working days and routes an authorised technician or advance RMA.",
        "Do not authorise third-party repair before Claim ID approval — unauthorised repair is a warranty exclusion.",
    ]:
        document.add_paragraph(step, style="List Number")

    document.add_heading("6. Unauthorised repair interaction", level=2)
    document.add_paragraph(
        "If a unit is within the warranty period but was opened by a local non-Pearlz technician, "
        "standard warranty coverage for that failure is declined under Warranty Policy §3.2. "
        "Paid service under AMC or time-and-materials remains available."
    )

    document.save(path)
    print(f"Wrote {path}")


def main() -> None:
    KB.mkdir(parents=True, exist_ok=True)
    write_product_manual()
    write_installation_guide()
    write_service_agreement()


if __name__ == "__main__":
    main()
