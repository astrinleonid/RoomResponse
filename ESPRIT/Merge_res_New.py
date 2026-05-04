import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Настройки путей
# ---------------------------------------------------------------------------

# Папка, в которой лежит сам скрипт
directory = os.path.dirname(os.path.abspath(__file__))
print("Рабочая папка:", directory)

# Диапазоны частот и ожидаемые файлы: results_{low}_{high}.json
band_ranges = [
    (40, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
]

# Максимальное количество мод
MAX_MODES = 128

# Число приёмников (как в исходном скрипте было 16)
NUM_RECEIVERS = 16

# ---------------------------------------------------------------------------
# Загрузка всех JSON файлов по диапазонам
# ---------------------------------------------------------------------------

all_data = {}

for low, high in band_ranges:
    file_name = f"results_{low}_{high}.json"
    file_path = os.path.join(directory, file_name)

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        all_data[(low, high)] = data
        print(f"Loaded {file_name}")
    else:
        print(f"File {file_name} not found in {directory}, skipping this band.")

if not all_data:
    print("No results files found, nothing to stitch.")
    raise SystemExit

# ---------------------------------------------------------------------------
# Сшивание данных: сбор всех мод из всех диапазонов
# ---------------------------------------------------------------------------

all_common_f = []
all_common_z = []
all_signed_shapes = []
all_participation = []
all_amplitudes_in_m = []

all_names = None
all_selected_r_indices = None

for band, data in all_data.items():
    # Частоты и затухания
    all_common_f.extend(data["common_f"])
    all_common_z.extend(data["common_z"])

    # Формы мод в узлах
    all_signed_shapes.extend([np.asarray(s, dtype=float) for s in data["signed_shapes"]])

    # Participation
    all_participation.extend([np.asarray(p, dtype=float) for p in data["participation"]])

    # Амплитуды в приёмниках (complex)
    all_amplitudes_in_m.extend(
        [[complex(c["real"], c["imag"]) for c in mode_ampl]
         for mode_ampl in data["amplitudes_in_m"]]
    )

    # предполагаем, что names и selected_r_indices одинаковые во всех файлах
    if all_names is None:
        all_names = data["names"]
    if all_selected_r_indices is None:
        all_selected_r_indices = data["selected_r_indices"]

# ---------------------------------------------------------------------------
# Сортировка по частоте и удаление дубликатов мод
# ---------------------------------------------------------------------------

# Сортируем по частоте
sort_idx = np.argsort(all_common_f)
all_common_f = [all_common_f[i] for i in sort_idx]
all_common_z = [all_common_z[i] for i in sort_idx]
all_signed_shapes = [all_signed_shapes[i] for i in sort_idx]
all_participation = [all_participation[i] for i in sort_idx]
all_amplitudes_in_m = [all_amplitudes_in_m[i] for i in sort_idx]

# Удаляем дубликаты частот (одинаковые частоты считаем одной модой)
freq_tol = 1e-6  # допуск по частоте

unique_idx = []
last_f = None
for i, f in enumerate(all_common_f):
    if last_f is None or abs(f - last_f) > freq_tol:
        unique_idx.append(i)
        last_f = f
    else:
        # дубликат – пропускаем
        pass

all_common_f        = [all_common_f[i]        for i in unique_idx]
all_common_z        = [all_common_z[i]        for i in unique_idx]
all_signed_shapes   = [all_signed_shapes[i]   for i in unique_idx]
all_participation   = [all_participation[i]   for i in unique_idx]
all_amplitudes_in_m = [all_amplitudes_in_m[i] for i in unique_idx]

num_modes = len(all_common_f)
print(f"After stitching and deduplication: {num_modes} modes.")
print(f"Frequency range: {all_common_f[0]:.2f} ... {all_common_f[-1]:.2f} Hz")

# ---------------------------------------------------------------------------
# Нормировка форм мод по энергии (по амплитудам на приёмниках)
# ---------------------------------------------------------------------------

mode_energies = [np.sum(np.abs(np.asarray(a)) ** 2) for a in all_amplitudes_in_m]
total_energy = float(np.sum(mode_energies))

if total_energy > 0:
    for k in range(num_modes):
        energy_contrib = mode_energies[k] / total_energy
        scale = np.sqrt(energy_contrib)
        all_signed_shapes[k] = all_signed_shapes[k] * scale

print(f"Total energy over all modes: {total_energy:.6f}")

# ---------------------------------------------------------------------------
# Подготовка сетки по узлам и интерполяция форм мод
# ---------------------------------------------------------------------------

# names – номера нот (узлов) как строки, приводим к int
note_numbers = [int(name) for name in all_names]
min_note = 0
max_note = max(note_numbers)

# Полная сетка узлов
full_notes = np.arange(min_note, max_note + 1)

# Интерполяция каждой моды на полную сетку
interpolated_shapes = []
for shape in all_signed_shapes:
    interp_func = interp1d(
        note_numbers,
        shape,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=False,
    )
    interpolated_shapes.append(interp_func(full_notes))

# ---------------------------------------------------------------------------
# Паддинг до MAX_MODES мод (если мод меньше)
# ---------------------------------------------------------------------------

if num_modes == 0:
    print("No modes after stitching, abort.")
    raise SystemExit

while len(all_common_f) < MAX_MODES:
    all_common_f.append(1000.0)                        # частота-заглушка
    all_common_z.append(1.0)                           # zeta-заглушка
    all_signed_shapes.append(np.zeros_like(all_signed_shapes[0]))
    all_participation.append(np.zeros_like(all_participation[0]))
    all_amplitudes_in_m.append([0j] * len(all_amplitudes_in_m[0]))
    interpolated_shapes.append(np.zeros_like(interpolated_shapes[0]))

effective_num_modes = num_modes  # реально посчитанные (до паддинга)

# ---------------------------------------------------------------------------
# Вычисление добротностей Q по zeta
# ---------------------------------------------------------------------------

BIGQ = 1e6
all_Q = [1.0 / (2.0 * z) if z > 0 else BIGQ for z in all_common_z]

# ---------------------------------------------------------------------------
# Ci_coef_cos.txt
# ---------------------------------------------------------------------------
# nn = 0..43 (итого 44), x = 2*nn + ni, ni = 0,1
# формируем массив [44 x 256], дальше разворачиваем в один столбец
# ---------------------------------------------------------------------------

NUM_NN = 44
NUM_ROWS = 256

ci_coef_cos = np.zeros((NUM_NN, NUM_ROWS))

for k in range(MAX_MODES):           # по всем 128 модам
    for idx_x, x in enumerate(full_notes):
        nn = x // 2
        ni = x % 2
        if nn < NUM_NN:
            ci_coef_cos[nn, k + ni * 128] = interpolated_shapes[k][idx_x]

ci_flat = ci_coef_cos.ravel(order="C").reshape(-1, 1)
np.savetxt(
    os.path.join(directory, "Ci_coef_cos.txt"),
    ci_flat,
    fmt="%.6f",
    delimiter="\t",
)

# ---------------------------------------------------------------------------
# omega_coef.txt – частоты
# ---------------------------------------------------------------------------

omega_coef = np.asarray(all_common_f[:MAX_MODES]).reshape(-1, 1)
np.savetxt(
    os.path.join(directory, "omega_coef.txt"),
    omega_coef,
    fmt="%.6f",
    delimiter="\t",
)

# ---------------------------------------------------------------------------
# Q_coeff_Q.txt и Q_coeff_E.txt – Q и zeta
# ---------------------------------------------------------------------------

q_coeff_q = np.asarray(all_Q[:MAX_MODES]).reshape(-1, 1)
q_coeff_e = np.asarray(all_common_z[:MAX_MODES]).reshape(-1, 1)

np.savetxt(
    os.path.join(directory, "Q_coeff_Q.txt"),
    q_coeff_q,
    fmt="%.6f",
    delimiter="\t",
)
np.savetxt(
    os.path.join(directory, "Q_coeff_E.txt"),
    q_coeff_e,
    fmt="%.6f",
    delimiter="\t",
)

# ---------------------------------------------------------------------------
# decka_coeff.txt – амплитуды мод в приёмниках
# ---------------------------------------------------------------------------

# Если в JSON есть selected_r_indices – используем их,
# иначе можно руками прописать, как было у тебя:
# present_receivers = [0, 1, 3, 4, 5]
if all_selected_r_indices is not None:
    present_receivers = sorted(set(all_selected_r_indices))
else:
    present_receivers = list(range(NUM_RECEIVERS))

M_eff = len(present_receivers)

decka_coeff = np.zeros((MAX_MODES, NUM_RECEIVERS))

for k in range(effective_num_modes):       # только реальные моды
    amps = np.asarray(all_amplitudes_in_m[k], dtype=complex)

    if amps.size == 0:
        continue  # ничего не заполняем, остаются нули

    if np.any(amps != 0):
        ref_phase = np.angle(amps[np.argmax(np.abs(amps))])
        signed_amps = np.real(amps * np.exp(-1j * ref_phase))
    else:
        # все амплитуды нулевые, просто берём нули той же длины
        signed_amps = np.real(amps)

    # НЕ выходим за пределы signed_amps
    for me, rec_idx in enumerate(present_receivers):
        if me >= len(signed_amps):
            break  # для этой моды амплитуд меньше, чем известных приёмников
        if 0 <= rec_idx < NUM_RECEIVERS:
            decka_coeff[k, rec_idx] = signed_amps[me]

np.savetxt(
    os.path.join(directory, "decka_coeff.txt"),
    decka_coeff,
    fmt="%.6f",
    delimiter="\t",
)

# ---------------------------------------------------------------------------
# Сохранение объединённых результатов в stitched_results.json
# ---------------------------------------------------------------------------

stitched_results = {
    "common_f": all_common_f[:MAX_MODES],
    "common_z": all_common_z[:MAX_MODES],
    "signed_shapes_interpolated": [s.tolist() for s in interpolated_shapes[:MAX_MODES]],
    "full_notes": full_notes.tolist(),
    "original_note_numbers": note_numbers,
    "participation": [p.tolist() for p in all_participation[:MAX_MODES]],
    "amplitudes_in_m": [
        [{"real": c.real, "imag": c.imag} for c in mode_ampl]
        for mode_ampl in all_amplitudes_in_m[:MAX_MODES]
    ],
    "names": all_names,
    "selected_r_indices": all_selected_r_indices,
}

output_path = os.path.join(directory, "stitched_results.json")
with open(output_path, "w") as f:
    json.dump(stitched_results, f, indent=4)

print(f"Stitched results saved to {output_path}")

# ---------------------------------------------------------------------------
# Интерактивный просмотр мод (стрелками ← / →)
# ---------------------------------------------------------------------------


class ModeViewer:
    def __init__(self, common_f, interpolated_shapes, full_notes, original_note_numbers):
        self.common_f = common_f
        self.interpolated_shapes = interpolated_shapes
        self.full_notes = full_notes
        self.original_note_numbers = np.asarray(original_note_numbers)
        self.k = 0

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_plot()

    def on_key(self, event):
        if event.key == "right":
            self.k = (self.k + 1) % len(self.common_f)
        elif event.key == "left":
            self.k = (self.k - 1) % len(self.common_f)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()

        # линия – интерполированная форма
        self.ax.plot(
            self.full_notes,
            self.interpolated_shapes[self.k],
            label="Interpolated shape",
        )

        # исходные узлы (синие) и чисто интерполированные (красные)
        is_original = np.isin(self.full_notes, self.original_note_numbers)
        self.ax.scatter(
            self.full_notes[is_original],
            self.interpolated_shapes[self.k][is_original],
            label="Original points",
        )
        self.ax.scatter(
            self.full_notes[~is_original],
            self.interpolated_shapes[self.k][~is_original],
            label="Interpolated-only points",
        )

        self.ax.set_xlabel("Note numbers")
        self.ax.set_ylabel("Normalized amplitude")
        self.ax.set_title(
            f"Mode {self.k}: f = {self.common_f[self.k]:.2f} Hz – signed shape along bridge"
        )
        self.ax.legend()
        self.fig.canvas.draw()


if len(all_common_f) > 0:
    viewer = ModeViewer(
        all_common_f[:MAX_MODES],
        interpolated_shapes[:MAX_MODES],
        full_notes,
        note_numbers,
    )
    plt.show()
else:
    print("No modes to display.")
