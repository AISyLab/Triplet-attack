import numpy as np
import h5py
from scipy import stats
import random

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

AES_Sbox_inv = np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

def labelize(plaintexts, keys):
    #return AES_Sbox[plaintexts[:, 2] ^ keys[:, 2]]
    state = [int(x) ^ int(k) for x, k in zip(np.asarray(plaintexts[:, 2]), keys[:, 2])]
    return AES_Sbox[state]

def addGussianNoise(traces, noise_level):
    print('Add Gussian noise: ', noise_level)
    if noise_level == 0:
        return traces
    else:
        output_traces = np.zeros(np.shape(traces))
        for trace in range(len(traces)):
            if(trace % 5000 == 0):
                print(str(trace) + '/' + str(len(traces)))
            profile_trace = traces[trace]
            noise = np.random.normal(
                0, noise_level, size=np.shape(profile_trace))
            output_traces[trace] = profile_trace + noise
        return output_traces

def addDesync(traces, desync_level):
    print('Add desync noise...')
    traces_length = len(traces[0])
    if desync_level == 0:
        return traces
    else:
        output_traces = np.zeros((len(traces), traces_length-desync_level))
        for idx in range(len(traces)):
            if(idx % 2000 == 0):
                print(str(idx) + '/' + str(len(traces)))
            rand = np.random.randint(low=0, high=desync_level)
            output_traces[idx] = traces[idx][rand:rand+traces_length-desync_level]
        return output_traces

def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return np.array([hw[int(s)] for s in data])

def load_ascad(ascad_database_file, profiling_traces=50000, leakage_model='HW'):
    if profiling_traces == 0:
        profiling_traces = 50000
    train_begin = 0
    train_end = profiling_traces
    test_begin = 0
    test_end  = 10000
    in_file = h5py.File(ascad_database_file + 'Base_desync0_4000.h5', "r")
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)[train_begin:train_end]
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    Y_profiling = np.array(in_file['Profiling_traces/labels'], dtype=np.int16)[train_begin:train_end]
    #Y_profiling = np.array(labelize(in_file['Profiling_traces/metadata']['plaintext'], in_file['Profiling_traces/metadata']['key']), dtype=np.int16)
    if leakage_model == 'HW':
        num_classes = 9
        Y_profiling = calculate_HW(Y_profiling)
    P_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'][:, 2], dtype=np.int16)[train_begin:train_end]
    
    # Load attack traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)[test_begin:test_end]
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    Y_attack = np.array(in_file['Attack_traces/labels'], dtype=np.int16)[test_begin:test_end]
    #Y_attack = np.array(labelize(in_file['Attack_traces/metadata']['plaintext'], in_file['Attack_traces/metadata']['key']), dtype=np.int16)
    if leakage_model == 'HW':
        Y_attack = calculate_HW(Y_attack)
    P_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][:, 2], dtype=np.int16)[test_begin:test_end]

    print('Profiling traces number: {}'.format(len(X_profiling)))
    print('Attack traces number: {}'.format(len(X_attack)))
    return (X_profiling, X_attack), (np.array(Y_profiling),  np.array(Y_attack)), (P_profiling,  P_attack)

def load_ascad_rand(ascad_database_file, profiling_traces=50000, leakage_model='HW'):
    if profiling_traces == 0:
        profiling_traces = 50000
    train_begin = 0
    train_end = profiling_traces
    test_begin = 0
    test_end  = 10000
    in_file = h5py.File(ascad_database_file + 'ascad-variable_4000.h5', "r")
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)[train_begin:train_end]
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    Y_profiling = np.array(in_file['Profiling_traces/labels'], dtype=np.int16)[train_begin:train_end]
    #Y_profiling = np.array(labelize(in_file['Profiling_traces/metadata']['plaintext'], in_file['Profiling_traces/metadata']['key']), dtype=np.int16)
    if leakage_model == 'HW':
        Y_profiling = calculate_HW(Y_profiling)
    P_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'][:, 2], dtype=np.int16)[train_begin:train_end]
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)[test_begin:test_end]
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    Y_attack = np.array(in_file['Attack_traces/labels'], dtype=np.int16)[test_begin:test_end]
    #Y_attack = np.array(labelize(in_file['Attack_traces/metadata']['plaintext'], in_file['Attack_traces/metadata']['key']), dtype=np.int16)
    if leakage_model == 'HW':
        Y_attack = calculate_HW(Y_attack)
    P_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][:, 2], dtype=np.int16)[test_begin:test_end]
    print('Profiling traces number: {}'.format(len(X_profiling)))
    print('Attack traces number: {}'.format(len(X_attack)))
    return (X_profiling, X_attack), (np.array(Y_profiling),  np.array(Y_attack)), (P_profiling,  P_attack)

def load_aeshd(AESHD_data_folder, profiling_traces=45000, leakage_model='HW', target_byte=11):
    def xor_bytes(a, b):
        return [a[i] ^ b[i] for i in range(len(a))]


    def expand_key(master_key):
        iteration_count = 0
        for i in range(4, 44):
            word = list(master_key[len(master_key) - 4:])

            if i % 4 == 0:
                word.append(word.pop(0))
                word = [AES_Sbox[b] for b in word]
                word[0] ^= r_con[i // 4]

            word = xor_bytes(word, master_key[iteration_count * 4:iteration_count * 4 + 4])
            for w in word:
                master_key.append(w)

            iteration_count += 1

        return [master_key[16 * i: 16 * (i + 1)] for i in range(len(master_key) // 16)]


    def get_round_keys(keys):
        """ Compute round keys for all keys" """

        keys = np.array(keys, dtype='uint8')
        if np.all(keys == keys[0]):
            """ If all keys are equal, then compute round keys for one key only """
            round_keys = expand_key(list(keys[0]))
            return np.full([len(keys), len(round_keys), len(round_keys[0])], round_keys)
        else:
            return [expand_key(list(key)) for key in keys]


    def get_labels_from_output(ciphertexts, keys, byte):
        """
        Labels for Hammind Distance leakage model: HD(InvSbox(c[i] xor k[i]), c[j]) = InvSbox(c[i] xor k[i]) xor c[j]
        k[i] = target key byte i of round key 10
        c[i] = ciphertext i
        c[j] = ciphertext j (j is different from i because of ShiftRows)
        """

        """ Compute round keys """
        round_keys = get_round_keys(keys)
        """ get ciphertext bytes c[i] and c[j]"""
        c_j = [cj[shift_row_mask[byte]] for cj in ciphertexts]
        c_i = [ci[byte] for ci in ciphertexts]
        """ get key byte from round key 10 """
        k_i = [ki[10][byte] for ki in round_keys]
        return [AES_Sbox_inv[int(ci) ^ int(ki)] ^ int(cj) for ci, cj, ki in zip(np.asarray(c_i[:]), np.asarray(c_j[:]), np.asarray(k_i[:]))]
    
    shift_row_mask = np.array([0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11])
    r_con = (
        0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
        0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
        0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
        0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
    )
    if profiling_traces == 0:
        profiling_traces = 45000

    in_file = h5py.File(AESHD_data_folder + 'aes_hd.h5', "r")

    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    profiling_key = in_file['Profiling_traces/metadata']['key']
    attack_key = in_file['Attack_traces/metadata']['key']
    # profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
    # attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
    C_profiling = np.array(in_file['Profiling_traces/metadata']['ciphertext'], dtype=np.int16)
    C_attack = np.array(in_file['Attack_traces/metadata']['ciphertext'], dtype=np.int16)
    Y_profiling = np.array(get_labels_from_output(C_profiling, profiling_key, target_byte), dtype=np.int16)
    Y_attack = np.array(get_labels_from_output(C_attack, attack_key, target_byte), dtype=np.int16)
    if leakage_model == 'HW':
        Y_profiling = calculate_HW(Y_profiling)
        Y_attack = calculate_HW(Y_attack)
    return (X_profiling[:profiling_traces], X_attack), (Y_profiling[:profiling_traces],  Y_attack), (C_profiling,  C_attack)
