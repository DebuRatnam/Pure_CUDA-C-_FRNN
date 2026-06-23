#pragma once

// Convert a runtime array index into a compile-time switch-case so NVCC can
// keep small thread-local arrays in registers rather than local memory.
// Local memory (L1-backed, ~100 cycle latency) vs registers (~4 cycles) is a
// 25× gap — critical for the K-NN scratch arrays accessed on every candidate.
//
// Usage:  RegisterIndexUtils<float, 16>::get(arr, runtime_k)
//         RegisterIndexUtils<float, 16>::set(arr, runtime_k, val)
//
// Supports N up to 128 (covers CAP ∈ {16, 32, 64, 128}).
template <typename T, int N> struct RegisterIndexUtils {
  __device__ __forceinline__ static T get(const T arr[N], int idx) {
    switch (idx) {
    case   0: return arr[  0]; case   1: return arr[  1];
    case   2: return arr[  2]; case   3: return arr[  3];
    case   4: return arr[  4]; case   5: return arr[  5];
    case   6: return arr[  6]; case   7: return arr[  7];
    case   8: return arr[  8]; case   9: return arr[  9];
    case  10: return arr[ 10]; case  11: return arr[ 11];
    case  12: return arr[ 12]; case  13: return arr[ 13];
    case  14: return arr[ 14]; case  15: return arr[ 15];
    case  16: return arr[ 16]; case  17: return arr[ 17];
    case  18: return arr[ 18]; case  19: return arr[ 19];
    case  20: return arr[ 20]; case  21: return arr[ 21];
    case  22: return arr[ 22]; case  23: return arr[ 23];
    case  24: return arr[ 24]; case  25: return arr[ 25];
    case  26: return arr[ 26]; case  27: return arr[ 27];
    case  28: return arr[ 28]; case  29: return arr[ 29];
    case  30: return arr[ 30]; case  31: return arr[ 31];
    case  32: return arr[ 32]; case  33: return arr[ 33];
    case  34: return arr[ 34]; case  35: return arr[ 35];
    case  36: return arr[ 36]; case  37: return arr[ 37];
    case  38: return arr[ 38]; case  39: return arr[ 39];
    case  40: return arr[ 40]; case  41: return arr[ 41];
    case  42: return arr[ 42]; case  43: return arr[ 43];
    case  44: return arr[ 44]; case  45: return arr[ 45];
    case  46: return arr[ 46]; case  47: return arr[ 47];
    case  48: return arr[ 48]; case  49: return arr[ 49];
    case  50: return arr[ 50]; case  51: return arr[ 51];
    case  52: return arr[ 52]; case  53: return arr[ 53];
    case  54: return arr[ 54]; case  55: return arr[ 55];
    case  56: return arr[ 56]; case  57: return arr[ 57];
    case  58: return arr[ 58]; case  59: return arr[ 59];
    case  60: return arr[ 60]; case  61: return arr[ 61];
    case  62: return arr[ 62]; case  63: return arr[ 63];
    case  64: return arr[ 64]; case  65: return arr[ 65];
    case  66: return arr[ 66]; case  67: return arr[ 67];
    case  68: return arr[ 68]; case  69: return arr[ 69];
    case  70: return arr[ 70]; case  71: return arr[ 71];
    case  72: return arr[ 72]; case  73: return arr[ 73];
    case  74: return arr[ 74]; case  75: return arr[ 75];
    case  76: return arr[ 76]; case  77: return arr[ 77];
    case  78: return arr[ 78]; case  79: return arr[ 79];
    case  80: return arr[ 80]; case  81: return arr[ 81];
    case  82: return arr[ 82]; case  83: return arr[ 83];
    case  84: return arr[ 84]; case  85: return arr[ 85];
    case  86: return arr[ 86]; case  87: return arr[ 87];
    case  88: return arr[ 88]; case  89: return arr[ 89];
    case  90: return arr[ 90]; case  91: return arr[ 91];
    case  92: return arr[ 92]; case  93: return arr[ 93];
    case  94: return arr[ 94]; case  95: return arr[ 95];
    case  96: return arr[ 96]; case  97: return arr[ 97];
    case  98: return arr[ 98]; case  99: return arr[ 99];
    case 100: return arr[100]; case 101: return arr[101];
    case 102: return arr[102]; case 103: return arr[103];
    case 104: return arr[104]; case 105: return arr[105];
    case 106: return arr[106]; case 107: return arr[107];
    case 108: return arr[108]; case 109: return arr[109];
    case 110: return arr[110]; case 111: return arr[111];
    case 112: return arr[112]; case 113: return arr[113];
    case 114: return arr[114]; case 115: return arr[115];
    case 116: return arr[116]; case 117: return arr[117];
    case 118: return arr[118]; case 119: return arr[119];
    case 120: return arr[120]; case 121: return arr[121];
    case 122: return arr[122]; case 123: return arr[123];
    case 124: return arr[124]; case 125: return arr[125];
    case 126: return arr[126]; case 127: return arr[127];
    default:  return T();
    }
  }

  __device__ __forceinline__ static void set(T arr[N], int idx, T val) {
    switch (idx) {
    case   0: arr[  0] = val; break; case   1: arr[  1] = val; break;
    case   2: arr[  2] = val; break; case   3: arr[  3] = val; break;
    case   4: arr[  4] = val; break; case   5: arr[  5] = val; break;
    case   6: arr[  6] = val; break; case   7: arr[  7] = val; break;
    case   8: arr[  8] = val; break; case   9: arr[  9] = val; break;
    case  10: arr[ 10] = val; break; case  11: arr[ 11] = val; break;
    case  12: arr[ 12] = val; break; case  13: arr[ 13] = val; break;
    case  14: arr[ 14] = val; break; case  15: arr[ 15] = val; break;
    case  16: arr[ 16] = val; break; case  17: arr[ 17] = val; break;
    case  18: arr[ 18] = val; break; case  19: arr[ 19] = val; break;
    case  20: arr[ 20] = val; break; case  21: arr[ 21] = val; break;
    case  22: arr[ 22] = val; break; case  23: arr[ 23] = val; break;
    case  24: arr[ 24] = val; break; case  25: arr[ 25] = val; break;
    case  26: arr[ 26] = val; break; case  27: arr[ 27] = val; break;
    case  28: arr[ 28] = val; break; case  29: arr[ 29] = val; break;
    case  30: arr[ 30] = val; break; case  31: arr[ 31] = val; break;
    case  32: arr[ 32] = val; break; case  33: arr[ 33] = val; break;
    case  34: arr[ 34] = val; break; case  35: arr[ 35] = val; break;
    case  36: arr[ 36] = val; break; case  37: arr[ 37] = val; break;
    case  38: arr[ 38] = val; break; case  39: arr[ 39] = val; break;
    case  40: arr[ 40] = val; break; case  41: arr[ 41] = val; break;
    case  42: arr[ 42] = val; break; case  43: arr[ 43] = val; break;
    case  44: arr[ 44] = val; break; case  45: arr[ 45] = val; break;
    case  46: arr[ 46] = val; break; case  47: arr[ 47] = val; break;
    case  48: arr[ 48] = val; break; case  49: arr[ 49] = val; break;
    case  50: arr[ 50] = val; break; case  51: arr[ 51] = val; break;
    case  52: arr[ 52] = val; break; case  53: arr[ 53] = val; break;
    case  54: arr[ 54] = val; break; case  55: arr[ 55] = val; break;
    case  56: arr[ 56] = val; break; case  57: arr[ 57] = val; break;
    case  58: arr[ 58] = val; break; case  59: arr[ 59] = val; break;
    case  60: arr[ 60] = val; break; case  61: arr[ 61] = val; break;
    case  62: arr[ 62] = val; break; case  63: arr[ 63] = val; break;
    case  64: arr[ 64] = val; break; case  65: arr[ 65] = val; break;
    case  66: arr[ 66] = val; break; case  67: arr[ 67] = val; break;
    case  68: arr[ 68] = val; break; case  69: arr[ 69] = val; break;
    case  70: arr[ 70] = val; break; case  71: arr[ 71] = val; break;
    case  72: arr[ 72] = val; break; case  73: arr[ 73] = val; break;
    case  74: arr[ 74] = val; break; case  75: arr[ 75] = val; break;
    case  76: arr[ 76] = val; break; case  77: arr[ 77] = val; break;
    case  78: arr[ 78] = val; break; case  79: arr[ 79] = val; break;
    case  80: arr[ 80] = val; break; case  81: arr[ 81] = val; break;
    case  82: arr[ 82] = val; break; case  83: arr[ 83] = val; break;
    case  84: arr[ 84] = val; break; case  85: arr[ 85] = val; break;
    case  86: arr[ 86] = val; break; case  87: arr[ 87] = val; break;
    case  88: arr[ 88] = val; break; case  89: arr[ 89] = val; break;
    case  90: arr[ 90] = val; break; case  91: arr[ 91] = val; break;
    case  92: arr[ 92] = val; break; case  93: arr[ 93] = val; break;
    case  94: arr[ 94] = val; break; case  95: arr[ 95] = val; break;
    case  96: arr[ 96] = val; break; case  97: arr[ 97] = val; break;
    case  98: arr[ 98] = val; break; case  99: arr[ 99] = val; break;
    case 100: arr[100] = val; break; case 101: arr[101] = val; break;
    case 102: arr[102] = val; break; case 103: arr[103] = val; break;
    case 104: arr[104] = val; break; case 105: arr[105] = val; break;
    case 106: arr[106] = val; break; case 107: arr[107] = val; break;
    case 108: arr[108] = val; break; case 109: arr[109] = val; break;
    case 110: arr[110] = val; break; case 111: arr[111] = val; break;
    case 112: arr[112] = val; break; case 113: arr[113] = val; break;
    case 114: arr[114] = val; break; case 115: arr[115] = val; break;
    case 116: arr[116] = val; break; case 117: arr[117] = val; break;
    case 118: arr[118] = val; break; case 119: arr[119] = val; break;
    case 120: arr[120] = val; break; case 121: arr[121] = val; break;
    case 122: arr[122] = val; break; case 123: arr[123] = val; break;
    case 124: arr[124] = val; break; case 125: arr[125] = val; break;
    case 126: arr[126] = val; break; case 127: arr[127] = val; break;
    }
  }
};
