#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace cpu
{
    // Demande au kernel de placer [ptr, ptr+bytes) sur des transparent huge
    // pages (2MB au lieu de 4KB) : pour les grosses tables à accès aléatoires
    // (poids NNUE ~108MB, TT), ça divise par ~512 le nombre d'entrées dTLB
    // nécessaires. MADV_HUGEPAGE couvre les pages pas encore touchées (et les
    // machines en mode THP "madvise", le défaut Ubuntu) ; MADV_COLLAPSE
    // (kernel >= 6.1) convertit immédiatement les pages déjà touchées --
    // nécessaire ici car make_shared/make_unique remplissent la mémoire
    // avant qu'on puisse la conseiller. Pur best-effort : les erreurs sont
    // ignorées, hors Linux c'est un no-op.
    inline void advise_huge_pages(const void *ptr, std::size_t bytes)
    {
#if defined(__linux__)
        const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        const auto page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
        const std::uintptr_t begin = (addr + page - 1) & ~(page - 1);
        const std::uintptr_t end = (addr + bytes) & ~(page - 1);
        if (end <= begin)
            return;
        void *aligned = reinterpret_cast<void *>(begin);
        const std::size_t len = end - begin;
        madvise(aligned, len, MADV_HUGEPAGE);
#ifdef MADV_COLLAPSE
        madvise(aligned, len, MADV_COLLAPSE);
#endif
#else
        (void)ptr;
        (void)bytes;
#endif
    }
}
