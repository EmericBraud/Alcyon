#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Mapping d'un fichier en LECTURE SEULE, partage entre processus.
//
// Raison d'etre : Alcyon lisait ses 111 Mo de poids NNUE dans le tas, donc
// N instances detenaient N copies physiques distinctes. Mesure sur 192 coeurs
// avec 164 parties simultanees : Alcyon perdait 58 % de son NPS quand
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
        // Zone reellement projetee (differente de base/size pour la copie
        // alignee sur pages geantes).
        void *raw = nullptr;
        std::size_t raw_size = 0;
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

#if defined(__linux__) && defined(MADV_HUGEPAGE)
        // Pages geantes (2 Mo) pour les poids : la mise a jour de
        // l'accumulateur lit ~10 lignes eparpillees dans ~110 Mo a chaque coup,
        // et chaque ligne sur une page de 4 Ko distincte coute un defaut de TLB.
        // Les pages geantes transparentes ne s'appliquent qu'a la memoire
        // anonyme : on recopie le fichier dans une zone alignee sur 2 Mo.
        // ALCYON_NO_HUGEPAGES=1 garde le partage du fichier entre processus.
        if (!std::getenv("ALCYON_NO_HUGEPAGES"))
        {
            constexpr std::size_t Huge = std::size_t(2) << 20;
            const std::size_t raw_size = size + Huge;
            void *raw = ::mmap(nullptr, raw_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (raw != MAP_FAILED)
            {
                auto *aligned = reinterpret_cast<std::byte *>((reinterpret_cast<std::uintptr_t>(raw) + Huge - 1) & ~(Huge - 1));
                ::madvise(aligned, size, MADV_HUGEPAGE);
                std::memcpy(aligned, addr, size);
                ::munmap(addr, size);
                auto *m = new Mapping{aligned, size, raw, raw_size};
                return std::shared_ptr<const Mapping>(m, [](const Mapping *p) {
                    ::munmap(p->raw, p->raw_size);
                    delete p;
                });
            }
        }
#endif
        auto *m = new Mapping{static_cast<const std::byte *>(addr), size, addr, size};
        return std::shared_ptr<const Mapping>(m, [](const Mapping *p) {
            ::munmap(p->raw, p->raw_size);
            delete p;
        });
    }

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
