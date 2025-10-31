# Map real edges id to computational edges id
edge_id_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 106: 4, 119: 5, 124: 6, 126: 7, 129: 8, 137: 9, 140: 10, 141: 11, 142: 12, 145: 13, 148: 14, 149: 15, 150: 16, 154: 17, 157: 18, 159: 19, 160: 20, 164: 21, 165: 22, 167: 23, 168: 24, 169: 25, 170: 26, 171: 27, 172: 28, 177: 29, 178: 30, 184: 31, 185: 32, 186: 33, 190: 34, 191: 35, 195: 36, 199: 37, 202: 38, 204: 39, 207: 40, 208: 41, 211: 42, 212: 43, 213: 44, 215: 45, 217: 46, 221: 47, 222: 48, 223: 49, 257: 50, 258: 51, 259: 52, 261: 53, 262: 54, 263: 55, 264: 56, 265: 57, 295: 58, 298: 59, 311: 60, 315: 61, 318: 62, 319: 63, 324: 64, 325: 65, 329: 66, 330: 67, 331: 68, 332: 69, 338: 70, 339: 71, 344: 72, 345: 73, 347: 74, 349: 75, 350: 76, 351: 77, 354: 78, 364: 79, 365: 80, 369: 81, 375: 82, 376: 83, 377: 84, 378: 85, 381: 86, 382: 87, 383: 88, 384: 89, 385: 90, 387: 91, 388: 92, 390: 93, 394: 94, 395: 95, 398: 96, 399: 97, 402: 98, 405: 99, 406: 100, 411: 101, 417: 102, 422: 103, 423: 104, 424: 105, 425: 106, 426: 107, 427: 108, 428: 109, 430: 110, 431: 111, 433: 112, 434: 113, 435: 114, 436: 115, 437: 116, 439: 117, 441: 118, 445: 119, 448: 120, 450: 121, 451: 122, 453: 123}

## General info
# Final number of nodes: 207
# Final number of edges: 206
# Temporal info (inputs, predictions and target) aggregated per node

## Procedure
# 1) Map real edges id to computational edges id through edge_id_mapping
# 2) Clean original dataset (eliminate duplicates)
# 3a) Create new edges to create a connected graph
# 3b) Update temporal data with fake new temporal information (not computed on loss calculation)

## Outputs
# 1) predictions for each nodes (207) at prediction_window-steps in the future (combined for all test set)
# 2) targets data for same period
# 3) inputs data used for the predictions

