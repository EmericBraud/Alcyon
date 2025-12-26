#pragma once
#include "move_list.hpp"

// Drapeaux pour savoir quel type de score nous avons stocké
enum TTFlag : uint8_t
{
    TT_EXACT, // Score exact (compris entre alpha et beta)
    TT_ALPHA, // Borne supérieure (Upper bound) : le score est <= alpha
    TT_BETA   // Borne inférieure (Lower bound) : le score est >= beta
};

struct TTEntry
{
    uint64_t key; // Pour vérifier les collisions
    Move move;    // Le meilleur coup (très important pour le tri des coups plus tard)
    int score;    // Le score de la position
    int depth;    // La profondeur de la recherche pour ce score
    uint8_t flag; // Type de score (TTFlag)
};

class TranspositionTable
{
private:
    std::vector<TTEntry> table;
    size_t size;

    // Gestion des scores de mat pour qu'ils soient valides peu importe le tour en cours
    int score_to_tt(int score, int ply)
    {
        if (score > MATE_SCORE - 1000)
            return score + ply; // On ajoute le ply pour "neutraliser" la distance à la racine
        if (score < -MATE_SCORE + 1000)
            return score - ply;
        return score;
    }

    int score_from_tt(int score, int ply)
    {
        if (score > MATE_SCORE - 1000)
            return score - ply; // On retire le ply actuel pour retrouver la distance réelle depuis la racine
        if (score < -MATE_SCORE + 1000)
            return score + ply;
        return score;
    }

public:
    // Taille en MB (ex: 64MB)
    void resize(size_t mb_size)
    {
        size_t entry_count = (mb_size * 1024 * 1024) / sizeof(TTEntry);
        table.resize(entry_count);
        size = entry_count;
        clear();
    }

    void clear()
    {
        std::fill(table.begin(), table.end(), TTEntry{0, Move(), 0, 0, 0});
    }

    // Sauvegarder une entrée
    void store(uint64_t key, int depth, int ply, int score, uint8_t flag, Move move)
    {
        size_t index = key % size; // Ou (key & (size - 1)) si size est une puissance de 2

        // Stratégie de remplacement simple : "Always Replace" ou "Depth-preferred"
        // Ici on remplace si nouvelle profondeur >= ancienne profondeur OU si c'est une hash différente
        if (table[index].key != key || depth >= table[index].depth)
        {
            table[index].key = key;
            table[index].depth = depth;
            table[index].flag = flag;
            table[index].score = score_to_tt(score, ply);
            table[index].move = move;
        }
    }

    // Lire une entrée. Retourne true si une entrée valide est trouvée et UTILISABLE pour couper
    bool probe(uint64_t key, int depth, int ply, int alpha, int beta, int &return_score, Move &best_move)
    {
        size_t index = key % size;
        TTEntry &entry = table[index];

        if (entry.key == key)
        {
            // On récupère le meilleur coup même si la profondeur n'est pas suffisante
            // Cela servira au tri des coups (move ordering)
            best_move = entry.move;

            if (entry.depth >= depth)
            {
                int score = score_from_tt(entry.score, ply);

                if (entry.flag == TT_EXACT)
                {
                    return_score = score;
                    return true;
                }
                if (entry.flag == TT_ALPHA && score <= alpha)
                {
                    return_score = score;
                    return true;
                }
                if (entry.flag == TT_BETA && score >= beta)
                {
                    return_score = score;
                    return true;
                }
            }
        }
        return false;
    }

    // Juste pour récupérer le coup (pour le tri)
    Move get_move(uint64_t key)
    {
        size_t index = key % size;
        if (table[index].key == key)
            return table[index].move;
        return Move(); // Coup null
    }
};