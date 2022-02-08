import numpy as np
import h5py

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

def labelize(plaintexts, keys):
    #return AES_Sbox[plaintexts[:, 2] ^ keys[:, 2]]
    state = [int(x) ^ int(k) for x, k in zip(np.asarray(plaintexts[:, 2]), keys[:, 2])]
    return AES_Sbox[state]

# Define your own root   
out_file_root = '/Base_desync0_4000.h5'
in_file = h5py.File('/ATMega8515_raw_traces.h5', "r")

traces = in_file["traces"]
metadata = in_file["metadata"]

n_profiling = 50000
n_attack = 10000

ns_resample = 4000

profiling_samples = np.zeros((n_profiling, ns_resample))
attack_samples = np.zeros((n_attack, ns_resample))

fs = 44000
ns = 4000

for i in range(n_profiling):
    if i%2000 == 0:
        print('{}/{}'.format(i, n_profiling))
    profiling_samples[i] = traces[i][fs:fs + ns]

for i in range(n_attack):
    if i%2000 == 0:
        print('{}/{}'.format(i, n_attack))
    attack_samples[i] = traces[n_profiling+i][fs:fs + ns]

profiling_plaintext = np.zeros((n_profiling, 16))
profiling_ciphertext = np.zeros((n_profiling, 16))
profiling_key = np.zeros((n_profiling, 16))
profiling_masks = np.zeros((n_profiling, 16))
for i, j in enumerate(range(0, n_profiling)):
    profiling_plaintext[i] = metadata[j][0]
    profiling_ciphertext[i] = metadata[j][1]
    profiling_key[i] = metadata[j][2]
    profiling_masks[i] = metadata[j][3]

attack_plaintext = np.zeros((n_attack, 16))
attack_ciphertext = np.zeros((n_attack, 16))
attack_key = np.zeros((n_attack, 16))
attack_masks = np.zeros((n_attack, 16))
for i in range(n_profiling, n_profiling + n_attack):
    attack_plaintext[i - n_profiling] = metadata[i][0]
    attack_ciphertext[i - n_profiling] = metadata[i][1]
    attack_key[i - n_profiling] = metadata[i][2]
    attack_masks[i - n_profiling] = metadata[i][3]

profiling_index = [n for n in range(0, len(profiling_samples))]
attack_index = [n for n in range(0, len(attack_samples))]

labels_profiling = labelize(profiling_plaintext, profiling_key)
labels_attack = labelize(attack_plaintext, attack_key)

# out_file = h5py.File('D:/traces/ATMega8515_raw_traces_resampled.h5', 'w')
out_file = h5py.File(out_file_root, 'w')

# Create our HDF5 hierarchy in the output file:
# Profiling traces with their labels
# Attack traces with their labels
profiling_traces_group = out_file.create_group("Profiling_traces")
attack_traces_group = out_file.create_group("Attack_traces")
# Datasets in the groups
profiling_traces_group.create_dataset(name="traces", data=profiling_samples, dtype=profiling_samples.dtype)
attack_traces_group.create_dataset(name="traces", data=attack_samples, dtype=attack_samples.dtype)
# Labels in the groups
profiling_traces_group.create_dataset(name="labels", data=labels_profiling, dtype=labels_profiling.dtype)
attack_traces_group.create_dataset(name="labels", data=labels_attack, dtype=labels_attack.dtype)

# TODO: deal with the case where "ciphertext" entry is there
# Put the metadata (plaintexts, keys, ...) so that one can check the key rank
metadata_type_profiling = np.dtype([("plaintext", profiling_plaintext.dtype, (len(profiling_plaintext[0]),)),
                                    ("ciphertext", profiling_ciphertext.dtype, (len(profiling_ciphertext[0]),)),
                                    ("key", profiling_key.dtype, (len(profiling_key[0]),)),
                                    ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))
                                    ])
metadata_type_attack = np.dtype([("plaintext", attack_plaintext.dtype, (len(attack_plaintext[0]),)),
                                 ("ciphertext", attack_ciphertext.dtype, (len(attack_ciphertext[0]),)),
                                 ("key", attack_key.dtype, (len(attack_key[0]),)),
                                 ("masks", attack_masks.dtype, (len(attack_masks[0]),))
                                 ])

profiling_metadata = np.array([(profiling_plaintext[n], profiling_ciphertext[n], profiling_key[n], profiling_masks[n]) for n, k in
                               zip(profiling_index, range(0, len(profiling_samples)))], dtype=metadata_type_profiling)
profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

attack_metadata = np.array([(attack_plaintext[n], attack_ciphertext[n], attack_key[n], attack_masks[n]) for n, k in
                            zip(attack_index, range(0, len(attack_samples)))], dtype=metadata_type_attack)
attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

out_file.flush()
out_file.close()

in_file = h5py.File(out_file_root, "r")
profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
attack_samples = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
profiling_ciphertext = in_file['Profiling_traces/metadata']['ciphertext']
attack_ciphertext = in_file['Attack_traces/metadata']['ciphertext']
profiling_key = in_file['Profiling_traces/metadata']['key']
attack_key = in_file['Attack_traces/metadata']['key']
profiling_masks = in_file['Profiling_traces/metadata']['masks']
attack_masks = in_file['Attack_traces/metadata']['masks']

print(len(profiling_plaintext))
print(len(attack_plaintext))
print(profiling_ciphertext)
print(attack_ciphertext)
print(profiling_key)
print(attack_key)
print(profiling_masks)
print(attack_masks)
