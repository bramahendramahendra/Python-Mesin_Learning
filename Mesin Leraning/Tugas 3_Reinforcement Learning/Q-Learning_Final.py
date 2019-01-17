# -*- coding: utf-8 -*-
"""
Created on Tue Apr  24 19:00:34 2018

@author: Brama Hendra M(1301150031)
"""

import numpy as np_1301150031

rows=10
cols=10
with open('Data Tugas 3 RL.txt') as txt_1301150031:
   data_1301150031 = []
   for loop_1301150031 in range(0, rows):
      data_1301150031.append(list(map(int, txt_1301150031.readline().split()[:cols])))

# Matriks 1
Matriks_1_1301150031 = np_1301150031.matrix(data_1301150031)
print("Matriks 1 inisialisasi dari matriks pada file txt :")
print(Matriks_1_1301150031)
print(" ")

# Matriks 2
Matriks_2_1301150031 = np_1301150031.matrix(np_1301150031.zeros([10,10]))
print("Matriks 2 inisialisasi untuk proses training :")
print(Matriks_2_1301150031)
print(" ")

# Nilai Gamma dmana Gamma = 0,8 .
gamma_1301150031 = 0.8

# Initial state. (Biasanya secara dipilih acak)
initial_state_1301150031 = 1

# Fungsi ini mengembalikan semua tindakan yang tersedia dalam keadaan yang diberikan sebagai argumen
def return_aksi_1301150031(status_1301150031):
    baris_1301150031 = Matriks_1_1301150031[status_1301150031,]
    hasil_aksi_1301150031 = np_1301150031.where(baris_1301150031 >= -1)[1]
    return hasil_aksi_1301150031

# Mendapatkan tindakan yang tersedia dalam keadaan saat ini
nilai_aksi_1301150031 = return_aksi_1301150031(initial_state_1301150031) 

# Fungsi ini secara acak memilih tindakan yang akan dilakukan dalam jangkauan semua tindakan tersedia.
def pemilihan_acak_sample_1301150031(jarak_1301150031):
    hasil_1301150031 = int(np_1301150031.random.choice(jarak_1301150031,1))
    return hasil_1301150031

# Sampel berikutnya tindakan untuk dilakukan
sample_1301150031 = pemilihan_acak_sample_1301150031(nilai_aksi_1301150031)

# Fungsi ini pembaruan matriks Q menurut jalan yang dipilih dan algoritma Q-learning 
def update_1301150031(status_1301150031, sample_1301150031, gamma_1301150031):
    
    nilai_max_1301150031 = np_1301150031.where(Matriks_2_1301150031[sample_1301150031,] == np_1301150031.max(Matriks_2_1301150031[sample_1301150031,]))[1]

    if nilai_max_1301150031.shape[0] > 1:
        nilai_max_1301150031 = int(np_1301150031.random.choice(nilai_max_1301150031, size = 1))
    else:
        nilai_max_1301150031 = int(nilai_max_1301150031)
    hasil_max_1301150031 = Matriks_2_1301150031[sample_1301150031, nilai_max_1301150031]
    
    # Rumus Q-learning
    Matriks_2_1301150031[status_1301150031, sample_1301150031] = Matriks_1_1301150031[status_1301150031, sample_1301150031] + gamma_1301150031 * hasil_max_1301150031

# Update Matriks 2
update_1301150031(initial_state_1301150031,sample_1301150031,gamma_1301150031)

#-------------------------------------------------------------------------------
# Training

# Train lebih dari 10000 iterasi. (Kembali iterate proses di atas).
for i in range(10000):
    current_state_13001150031 = np_1301150031.random.randint(0, int(Matriks_2_1301150031.shape[0]))
    nilai_aksi_1301150031 = return_aksi_1301150031(current_state_13001150031)
    aksi_1301150031 = pemilihan_acak_sample_1301150031(nilai_aksi_1301150031)
    update_1301150031(current_state_13001150031,aksi_1301150031,gamma_1301150031)
    
# Normalize the "trained" Matriks 2
print("Trained Matriks:")
print(Matriks_2_1301150031/np_1301150031.max(Matriks_2_1301150031)*100)
print(" ") 

#-------------------------------------------------------------------------------
# Testing

# Goal state = 5

current_state_1301150031 = 2
hasil_steps_1301150031 = [current_state_1301150031]

while current_state_1301150031 != 5:

    step_selanjutnya_13011150031 = np_1301150031.where(Matriks_2_1301150031[current_state_1301150031,] == np_1301150031.max(Matriks_2_1301150031[current_state_1301150031,]))[1]
    
    if step_selanjutnya_13011150031.shape[0] > 1:
        step_selanjutnya_13011150031 = int(np_1301150031.random.choice(step_selanjutnya_13011150031, size = 1))
    else:
        step_selanjutnya_13011150031 = int(step_selanjutnya_13011150031)
    
    hasil_steps_1301150031.append(step_selanjutnya_13011150031)
    current_state_1301150031 = step_selanjutnya_13011150031

# Print pilihan dari langkah-langkah
print("Pilihan path:")
print(hasil_steps_1301150031)
