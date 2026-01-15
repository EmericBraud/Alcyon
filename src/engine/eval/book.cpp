#include "engine/eval/book.hpp"
#include "core/move/generator/move_generator.hpp"

#include "common/logger.hpp"

#include <random>
#include <fstream>

std::vector<Book::Entry> Book::entries;

void Book::init(const std::string &path)
{
    entries.clear();
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "!!! ERREUR BOOK: Impossible d'ouvrir le fichier : " << path << std::endl;
        return;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < sizeof(Entry))
        return;

    size_t num_entries = file_size / sizeof(Entry);
    entries.resize(num_entries);
    file.read(reinterpret_cast<char *>(entries.data()), file_size);
    file.close();

    // 1. Endianness conversion
    for (auto &e : entries)
    {
        e.key = swap_endian_64(e.key);
        e.move = swap_endian_16(e.move);
        e.weight = swap_endian_16(e.weight);
        e.learn = swap_endian_32(e.learn);
    }

    // 2. Dichotimie sort
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b)
              { return a.key < b.key; });

    logs::debug << ">>> SUCCES BOOK: " << num_entries << " coups chargés." << std::endl;
}

uint16_t Book::swap_endian_16(uint16_t val) { return (val << 8) | (val >> 8); }
uint32_t Book::swap_endian_32(uint32_t val)
{
    return ((val << 24) & 0xff000000) | ((val << 8) & 0x00ff0000) |
           ((val >> 8) & 0x0000ff00) | ((val >> 24) & 0x000000ff);
}
uint64_t Book::swap_endian_64(uint64_t val)
{
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL) | ((val >> 8) & 0x00FF00FF00FF00FFULL);
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL) | ((val >> 16) & 0x0000FFFF0000FFFFULL);
    return (val << 32) | (val >> 32);
}

Move Book::probe(Board &board)
{
    if (entries.empty())
        return Move();

    uint64_t key = board.polyglot_key();

    // 1. Recherche binaire
    auto it = std::lower_bound(entries.begin(), entries.end(), key,
                               [](const Entry &e, uint64_t k)
                               { return e.key < k; });

    // Si pas trouvé
    if (it == entries.end() || it->key != key)
        return Move();

    // 2. Récupérer les coups RAW (uint16_t) candidats
    std::vector<std::pair<uint16_t, int>> candidates;
    int total_weight = 0;

    for (; it != entries.end() && it->key == key; ++it)
    {
        if (it->weight > 0)
        {
            candidates.push_back({it->move, it->weight});
            total_weight += it->weight;
        }
    }

    if (candidates.empty())
        return Move();

    // 3. Sélection pondérée
    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, total_weight - 1);
    int pick = dist(gen);
    int current = 0;
    uint16_t selected_poly_move = 0;

    for (const auto &pair : candidates)
    {
        current += pair.second;
        if (pick < current)
        {
            selected_poly_move = pair.first;
            break;
        }
    }
    // Fallback
    if (selected_poly_move == 0)
        selected_poly_move = candidates.back().first;

    // =========================================================
    // 4. MATCHING AVEC LE GÉNÉRATEUR DU MOTEUR
    // =========================================================

    // Décodage Polyglot
    int target_to = (selected_poly_move >> 0) & 0x3F;   // <--- Bits 0-5 = TO
    int target_from = (selected_poly_move >> 6) & 0x3F; // <--- Bits 6-11 = FROM
    int promo_code = (selected_poly_move >> 12) & 0x7;

    Piece target_promo = NO_PIECE;
    switch (promo_code)
    {
    case 1:
        target_promo = KNIGHT;
        break;
    case 2:
        target_promo = BISHOP;
        break;
    case 3:
        target_promo = ROOK;
        break;
    case 4:
        target_promo = QUEEN;
        break;
    }

    MoveList moves;
    MoveGen::generate_legal_moves(board, moves);

    for (const auto &m : moves)
    {
        if (m.get_from_sq() == target_from && m.get_to_sq() == target_to)
        {

            if (target_promo != NO_PIECE)
            {
                if (m.is_promotion() && m.get_promo_piece() == target_promo)
                {
                    logs::debug << "info string DEBUG: Book Move Matches Engine Move: " << m.to_uci() << std::endl;
                    return m;
                }
            }
            else
            {
                logs::debug << "info string DEBUG: Book Move Matches Engine Move: " << m.to_uci() << std::endl;
                return m;
            }
        }
    }
    logs::debug << "DEBUG: Book Move (" << target_from << "->" << target_to
                << ") not found in engine !";

    return Move(); // Coup vide
}