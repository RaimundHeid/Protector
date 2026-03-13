Protector is a Bitboard-based chess program that communicates with a chess GUI via the UCI protocol.

Copyright (C) 2009-2010 Raimund Heid (Raimund_Heid@yahoo.com)

### TERMS OF USE

Protector is free, and distributed under the GNU General Public License(GPL). Essentially, this means that you are free to do almost what you want with the program, including distributing it among your friends, making it available for download from your web site, selling it (either by itself or as part of some bigger software package), or using it as the starting point for a software project of your own.

The only real limitation is that whenever you distribute Protector in some way, you must always include the full source code, or a pointer to where the source code can be found.  If you make any changes to the source code, these changes must also be made available under the GPL.

For full details, read the copy of the GPL found in the file named Copying.txt.

### CREDITS

Protector is based on many great ideas from the following people: Fabien Letouzey (pvnodes, blending of opening and endgame values, eval params), Thomas Gaksch (pvnode extensions, extended futility pruning, space attack eval), Robert Hyatt (consistent hashtable entries), Stefan Meyer-Kahlen (UCI), Gerd Isenberg/Lasse Hansen (magic bitboards), Marco Costabla/Tord Romstad/Joona Kiiski (Glaurung/Stockfish sources), Vasik Rajlich/Larry Kaufman (singlemove extensions, op/eg integer arithmetics, values for material imbalances in Rybka/Robbolito). Without their contributions Protector would not be what it is. Thank you so much.

### FATHOM

Protector integrates the Fathom library for Syzygy tablebase probing.

Fathom:
Copyright (c) 2013-2018 Ronald de Man
Copyright (c) 2015 basil00
Modifications Copyright (c) 2016-2024 by Jon Dart

Fathom is licensed under the MIT License. The source files for Fathom in the `src/` directory (including `tbprobe.c`, `tbchess.c`, `tbconfig.h`, and `tbprobe.h`) contain the full license text.