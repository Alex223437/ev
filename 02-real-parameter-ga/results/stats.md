| Funkce | Varianta | Nejlepší | Nejhorší | Průměr | Medián | Sm. odchylka | Nastavení |
|---|---|---|---|---|---|---|---|
| Sphere | IEEE 754 | 1.162e-86 | 3.345e-83 | 4.932e-84 | 4.684e-85 | 1.081e-83 | N=20, elite=20%, rank, pc=1, pm=0.003125 |
| Sphere | Fixed-point | 2.359e-16 | 2.452e-13 | 2.636e-14 | 1.096e-15 | 7.694e-14 | N=20, elite=20%, rank, pc=1, pm=0.003125 |
| Sphere | BCD | 2.634e-05 | 0.04016 | 0.004314 | 0.0003082 | 0.0126 | N=100, elite=20%, rank, pc=1, pm=0.01 |
| Sphere | Real + Gaussova mutace | 5.76e-06 | 3.785e-05 | 1.574e-05 | 1.151e-05 | 1.151e-05 | N=20, elite=20%, rank, pc=1, pm=0.1, sigma=0.01*rozsah |
| Sphere | Real + rovnoměrná mutace | 0.001611 | 0.01889 | 0.007761 | 0.006029 | 0.005348 | N=20, elite=20%, rank, pc=1, pm=0.1 |
| Rosenbrock | IEEE 754 | 6.194 | 8.962 | 8.508 | 8.727 | 0.8292 | N=100, elite=20%, rank, pc=1, pm=0.003125 |
| Rosenbrock | Fixed-point | 3.351 | 7.735 | 6.271 | 6.644 | 1.471 | N=100, elite=20%, rank, pc=0.8, pm=0.01 |
| Rosenbrock | BCD | 5.685 | 8.072 | 7.154 | 7.163 | 0.8802 | N=100, elite=20%, rank, pc=1, pm=0.01 |
| Rosenbrock | Real + Gaussova mutace | 0.1668 | 7.735 | 6.363 | 7.247 | 2.276 | N=20, elite=10%, rank, pc=1, pm=0.1, sigma=0.05*rozsah |
| Rosenbrock | Real + rovnoměrná mutace | 1.373 | 61.28 | 12.33 | 7.549 | 17.35 | N=100, elite=20%, rank, pc=1, pm=0.1 |
| Schwefel | IEEE 754 | 5.276 | 779.5 | 300.5 | 274.1 | 232.9 | N=100, elite=10%, roulette, pc=1, pm=0.003125 |
| Schwefel | Fixed-point | 1.551 | 134.2 | 44.69 | 17.46 | 54.34 | N=100, elite=20%, rank, pc=1, pm=0.03 |
| Schwefel | BCD | 55.94 | 686.5 | 264.1 | 237.3 | 179.7 | N=100, elite=20%, rank, pc=0.8, pm=0.01 |
| Schwefel | Real + Gaussova mutace | 4.229 | 242.8 | 71.6 | 24.74 | 81.41 | N=100, elite=20%, rank, pc=1, pm=0.1, sigma=0.3*rozsah |
| Schwefel | Real + rovnoměrná mutace | 5.066 | 17.07 | 10.54 | 10.68 | 4.234 | N=20, elite=20%, rank, pc=0.8, pm=0.1 |
