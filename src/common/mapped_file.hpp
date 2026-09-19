#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Mapping d'un fichier en LECTURE SEULE, partage entre processus.
//
// Raison d'etre : chess26 lisait ses 111 Mo de poids NNUE dans le tas, donc
// N instances detenaient N copies physiques distinctes. Mesure sur 192 coeurs
// avec 164 parties simultanees : chess26 perdait 58 % de son NPS quand
// Stockfish 8, dont l'eval tient en quelques kilo-octets, n'en perdait que
// 20 %. Un handicap relatif double, soit ~86 Elo de biais sur un match contre
// un moteur leger.
//
// MAP_SHARED + PROT_READ fait pointer les tables de pages de tous les
// processus sur les MEMES pages physiques (celles du cache de pages). Aucune
// ecriture, donc aucune copie sur ecriture : une seule copie de 111 Mo, et
// une ligne de cache chargee par un processus sert a tous les autres.
namespace mapped
{
    struct Mapping
    {
        const std::byte *base = nullptr;
        std::size_t size = 0;
    };

    // Renvoie un shared_ptr<Mapping> dont le deleter fait le munmap, ou
    // nullptr si le mapping echoue -- l'appelant retombe alors sur la lecture
    // classique. Jamais fatal : c'est une optimisation, pas une dependance.
    inline std::shared_ptr<const Mapping> map_readonly(const std::string &path)
    {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            return nullptr;

        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size <= 0)
        {
            ::close(fd);
            return nullptr;
        }

        const std::size_t size = static_cast<std::size_t>(st.st_size);
        void *addr = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd); // le mapping garde sa propre reference au fichier
        if (addr == MAP_FAILED)
            return nullptr;

        auto *m = new Mapping{static_cast<const std::byte *>(addr), size};
        return std::shared_ptr<const Mapping>(m, [](const Mapping *p) {
            ::munmap(const_cast<void *>(static_cast<const void *>(p->base)), p->size);
            delete p;
        });
    }

    // Vue typee sur une region du mapping, dont la duree de vie est portee
    // par le mapping lui-meme (constructeur aliasing de shared_ptr).
    //
    // ponytail: reinterpret_cast sur des octets mappes -- formellement il
    // faudrait std::start_lifetime_as (C++23). Aucun compilateur reel ne
    // pose probleme ici (T est trivialement copiable et la disposition
    // memoire est celle du fichier), mais si on passe a C++23, remplacer.
    template <typename T>
    std::shared_ptr<const T> view_at(const std::shared_ptr<const Mapping> &map, std::size_t offset)
    {
        if (!map || offset + sizeof(T) > map->size)
            return nullptr;
        const std::byte *p = map->base + offset;
        if (reinterpret_cast<std::uintptr_t>(p) % alignof(T) != 0)
            return nullptr; // desaligne : l'appelant retombe sur la copie
        return std::shared_ptr<const T>(map, reinterpret_cast<const T *>(p));
    }
}
