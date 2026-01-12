#pragma once
#include "core/board.hpp"

class Book
{
public:
    // Charge tout le fichier en mémoire au démarrage.
    // A appeler dans le main() !
    static void init(const std::string &path);

    // Recherche instantanée en mémoire
    static Move probe(Board &board);

private:
// Structure alignée sur 1 octet pour correspondre au binaire
#pragma pack(push, 1)
    struct Entry
    {
        uint64_t key;    // 8 octets
        uint16_t move;   // 2 octets
        uint16_t weight; // 2 octets
        uint32_t learn;  // 4 octets
    };
#pragma pack(pop)

    // Stockage de tout le livre en RAM
    static std::vector<Entry> entries;

    // Utilitaires endianness
    static uint16_t swap_endian_16(uint16_t val);
    static uint32_t swap_endian_32(uint32_t val);
    static uint64_t swap_endian_64(uint64_t val);

    static Move convert_poly_move_to_internal(uint16_t poly_move, const Board &board);
};