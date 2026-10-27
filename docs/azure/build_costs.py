"""Genera docs/azure/costos.xlsx (modelo de costos de la arquitectura Azure).

Precios: API pública de precios de Azure (prices.azure.com), EUR, región westeurope salvo Azure OpenAI
(swedencentral, despliegue Data Zone UE), consultados el 27 oct 2026. Los supuestos van en azul y se pueden
cambiar; todas las demás celdas son fórmulas.
Uso: python docs/azure/build_costs.py
"""
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

OUT = Path(__file__).with_name("costos.xlsx")
BLUE = Font(color="1F4FD8")
BOLD = Font(bold=True)
HEAD = PatternFill("solid", fgColor="E8EDF5")

SUPUESTOS = [
    # clave, descripción, valor, unidad, origen
    ("estaciones", "Estaciones de alineación", 1, "estaciones", "Enunciado: 1 operario por turno por estación"),
    ("piezas_min", "Piezas por minuto por estación", 20, "piezas/min", "Supuesto"),
    ("horas_semana", "Horas de operación por semana", 120, "h", "Enunciado: 24 h x 5 días"),
    ("semanas_mes", "Semanas por mes", 4.33, "", "Calendario"),
    ("kb_imagen", "Tamaño medio de imagen guardada", 30, "KB", "Supuesto (COIL-100 PNG ~15 KB)"),
    ("meses_caliente", "Meses de imágenes en nivel caliente", 3, "meses", "Diseño: 90 días"),
    ("tasa_desvio", "Fracción de piezas desviadas a revisión", 0.10, "", "Supuesto; ver reporte de visión"),
    ("seg_revision", "Segundos por revisión manual", 10, "s", "Supuesto"),
    ("consultas_dia", "Consultas al asistente por día", 200, "consultas", "Supuesto"),
    ("tok_in", "Tokens de entrada por consulta", 6000, "tokens", "Medido en la evaluación del agente (ajustar)"),
    ("tok_out", "Tokens de salida por consulta", 400, "tokens", "Medido en la evaluación del agente (ajustar)"),
    ("frac_cache", "Fracción de entrada servida desde caché", 0.5, "", "Supuesto"),
    ("pg_storage_gb", "Almacenamiento PostgreSQL", 32, "GB", "Supuesto"),
    ("logs_gb", "Ingesta de logs por mes", 5, "GB", "Supuesto"),
    ("horas_entreno", "Horas de entrenamiento al mes (D4s v5)", 10, "h", "Supuesto"),
    ("vcpu_agente", "vCPU del agente (Container Apps)", 1, "vCPU", "Diseño"),
    ("gib_agente", "Memoria del agente (Container Apps)", 2, "GiB", "Diseño"),
    ("operarios", "Operarios por estación (3 turnos)", 3, "personas", "Enunciado"),
    ("coste_operario", "Coste anual cargado por operario", 30000, "EUR/año", "Supuesto, ajustar al país"),
    ("horas_fte_mes", "Horas laborales por persona y mes", 160, "h", "Supuesto"),
    ("pc_edge", "PC industrial por estación (edge)", 2000, "EUR", "Supuesto"),
    ("amort_meses", "Amortización del PC edge", 36, "meses", "Supuesto"),
    ("capex_proyecto", "Coste de implantación del proyecto", 40000, "EUR", "Supuesto"),
]

PRECIOS = [
    ("p_iothub_s1", "IoT Hub S1", 21.9993, "EUR/unidad-mes", "westeurope"),
    ("p_ca_vcpu", "Container Apps Dedicated vCPU", 0.0712, "EUR/vCPU-h", "westeurope (cota superior; Consumption tiene tramo gratuito)"),
    ("p_ca_mem", "Container Apps Dedicated memoria", 0.0058, "EUR/GiB-h", "westeurope"),
    ("p_ca_plan", "Container Apps Dedicated gestión del plan", 0.088, "EUR/h", "westeurope"),
    ("p_aoai_in", "Azure OpenAI gpt-4.1-mini entrada (Data Zone)", 0.0004, "EUR/1K tokens", "swedencentral"),
    ("p_aoai_cached", "Azure OpenAI gpt-4.1-mini entrada en caché", 0.0001, "EUR/1K tokens", "swedencentral"),
    ("p_aoai_out", "Azure OpenAI gpt-4.1-mini salida (Data Zone)", 0.0015, "EUR/1K tokens", "swedencentral"),
    ("p_pg_b1ms", "PostgreSQL Flexible B1MS", 0.0175, "EUR/h", "westeurope"),
    ("p_pg_storage", "PostgreSQL Flexible almacenamiento", 0.1205, "EUR/GB-mes", "westeurope"),
    ("p_blob_hot", "Blob Hot LRS", 0.0172, "EUR/GB-mes", "westeurope"),
    ("p_blob_archive", "Blob Archive LRS", 0.0016, "EUR/GB-mes", "westeurope"),
    ("p_acr_basic", "Container Registry Basic", 0.1466, "EUR/día", "westeurope"),
    ("p_logs", "Log Analytics ingesta", 2.6311, "EUR/GB", "westeurope"),
    ("p_kv_ops", "Key Vault operaciones", 0.0264, "EUR/10K", "westeurope"),
    ("p_vm_d4", "VM D4s v5 (entrenamiento)", 0.2024, "EUR/h", "westeurope"),
    ("p_search_basic", "Azure AI Search Basic (alternativa)", 0.0889, "EUR/h", "westeurope"),
]


def header(ws, cols):
    for i, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=i, value=c)
        cell.font, cell.fill = BOLD, HEAD
    ws.freeze_panes = "A2"


def main():
    wb = Workbook()
    names = {}

    ws = wb.active
    ws.title = "Supuestos"
    header(ws, ["Clave", "Supuesto", "Valor", "Unidad", "Origen"])
    for r, (k, d, v, u, o) in enumerate(SUPUESTOS, 2):
        ws.append([k, d, v, u, o])
        ws.cell(row=r, column=3).font = BLUE
        names[k] = f"Supuestos!$C${r}"

    wp = wb.create_sheet("Precios")
    header(wp, ["Clave", "Concepto", "Precio", "Unidad", "Región / nota"])
    for r, row in enumerate(PRECIOS, 2):
        wp.append(list(row))
        names[row[0]] = f"Precios!$C${r}"
    wp.append([])
    wp.append(["", "Fuente: API pública de precios de Azure (prices.azure.com), EUR, consulta del 27 oct 2026."])

    n = names
    piezas_mes = f"({n['piezas_min']}*60*{n['horas_semana']}*{n['semanas_mes']}*{n['estaciones']})"
    horas_mes = "730"
    wc = wb.create_sheet("Costos_Azure")
    header(wc, ["Capa", "Servicio", "Cálculo", "EUR/mes", "Por qué este servicio"])
    filas = [
        ("Planta", "Azure IoT Edge en PC industrial", "Amortización del PC por estación",
         f"={n['pc_edge']}/{n['amort_meses']}*{n['estaciones']}", "Inferencia junto al robot: sin latencia de red y la línea sigue si cae internet"),
        ("Planta", "IoT Hub S1", "1 unidad", f"={n['p_iothub_s1']}", "Telemetría, comandos y despliegue del contenedor al edge"),
        ("Asistente", "Container Apps (agente)", "vCPU y memoria 24x7, plan Dedicated (cota superior)",
         f"=({n['vcpu_agente']}*{n['p_ca_vcpu']}+{n['gib_agente']}*{n['p_ca_mem']}+{n['p_ca_plan']})*{horas_mes}",
         "Sin clúster que operar; escala con eventos"),
        ("Asistente", "Azure OpenAI gpt-4.1-mini", "consultas x 30 x tokens x precio",
         f"={n['consultas_dia']}*30*({n['tok_in']}*(1-{n['frac_cache']})*{n['p_aoai_in']}+{n['tok_in']}*{n['frac_cache']}*{n['p_aoai_cached']}+{n['tok_out']}*{n['p_aoai_out']})/1000",
         "Residencia de datos en la UE y red privada"),
        ("Datos", "PostgreSQL Flexible B1MS", "cómputo 730 h + almacenamiento",
         f"={n['p_pg_b1ms']}*{horas_mes}+{n['pg_storage_gb']}*{n['p_pg_storage']}", "Casos, tickets, auditoría y búsqueda en SOP con pgvector"),
        ("Datos", "Blob Storage (imágenes)", "GB en caliente (3 meses) + archivo",
         f"={piezas_mes}*{n['kb_imagen']}/1048576*({n['meses_caliente']}*{n['p_blob_hot']}+{n['p_blob_archive']}*12)",
         "Niveles caliente y archivo con reglas de ciclo de vida"),
        ("ML", "Azure Machine Learning (cómputo)", "horas de entrenamiento D4s v5", f"={n['horas_entreno']}*{n['p_vm_d4']}",
         "Lee MLflow; registro de modelos y despliegue al edge"),
        ("Plataforma", "Container Registry Basic", "30 días", f"=30*{n['p_acr_basic']}", "Imágenes de visión y agente"),
        ("Plataforma", "Key Vault", "≈100K operaciones", f"=10*{n['p_kv_ops']}", "Secretos con identidades administradas"),
        ("Plataforma", "Log Analytics / App Insights", "GB de logs", f"={n['logs_gb']}*{n['p_logs']}", "Trazas del agente, tokens y latencia"),
    ]
    for f in filas:
        wc.append(list(f))
    last = len(filas) + 1
    wc.append([])
    wc.append(["", "Total Azure + edge", "", f"=SUM(D2:D{last})", ""])
    total_row = last + 2
    wc.cell(row=total_row, column=2).font = BOLD
    wc.cell(row=total_row, column=4).font = BOLD
    wc.append([])
    wc.append(["Alternativa", "Azure AI Search Basic", "730 h", f"={n['p_search_basic']}*{horas_mes}",
               "No incluida: coste fijo alto para decenas de SOP; usar cuando crezca el corpus"])
    names["total_azure"] = f"Costos_Azure!$D${total_row}"

    wcmp = wb.create_sheet("Comparativa")
    header(wcmp, ["Concepto", "EUR/mes", "Nota"])
    rows = [
        ("Operación actual: operarios de alineación", f"={n['operarios']}*{n['estaciones']}*{n['coste_operario']}/12", "3 turnos x 1 operario por estación"),
        ("Sistema: Azure + edge", f"={n['total_azure']}", "Hoja Costos_Azure"),
        ("Sistema: personas en revisión manual (horas)", f"={piezas_mes}*{n['tasa_desvio']}*{n['seg_revision']}/3600", "Horas al mes de revisión de desvíos"),
        ("Sistema: coste de revisión manual", f"=B4/{n['horas_fte_mes']}*{n['coste_operario']}/12", "Horas convertidas a personas a tiempo completo"),
        ("Sistema: total", "=B3+B5", ""),
        ("Ahorro mensual", "=B2-B6", ""),
        ("Meses para recuperar la implantación", f"=IF(B7>0,{n['capex_proyecto']}/B7,\"no se recupera\")", "Supuesto de implantación en Supuestos"),
    ]
    for r in rows:
        wcmp.append(list(r))
    wcmp.append([])
    wcmp.append(["Lectura", "", "Si la tasa de desvío sube, crece la revisión manual: es la palanca principal del caso de negocio."])

    for sheet in wb.worksheets:
        for col in range(1, sheet.max_column + 1):
            sheet.column_dimensions[get_column_letter(col)].width = 22 if col != 5 else 70
        for row in sheet.iter_rows():
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(OUT)
    print("guardado", OUT)


if __name__ == "__main__":
    main()
