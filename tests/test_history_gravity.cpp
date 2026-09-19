#include <gtest/gtest.h>
#include "engine/search/worker.hpp"

// La gravite doit borner les entrees sans clamp explicite : c'est la seule
// propriete dont depend l'ordonnancement (une table saturee ordonne au bruit).
TEST(HistoryGravity, StaysBoundedUnderRepeatedBonus)
{
    int e = 0;
    for (int i = 0; i < 10000; ++i)
        SearchWorker::update_hist(e, 400); // depth*depth pour depth=20
    EXPECT_LE(e, SearchWorker::HistMax); // point fixe = +HistMax exactement
    EXPECT_GT(e, 0);
}

TEST(HistoryGravity, StaysBoundedUnderRepeatedMalus)
{
    int e = 0;
    for (int i = 0; i < 10000; ++i)
        SearchWorker::update_hist(e, -400);
    EXPECT_GE(e, -SearchWorker::HistMax); // point fixe = -HistMax exactement
    EXPECT_LT(e, 0);
}

// Un bonus enorme ne doit pas faire exploser l'entree : il est clampe avant
// d'etre applique, et la gravite le ramene dans le domaine.
TEST(HistoryGravity, SaturatingBonusIsAbsorbed)
{
    int e = 0;
    SearchWorker::update_hist(e, 1 << 20);
    EXPECT_LE(std::abs(e), SearchWorker::HistMax);
}

// Une entree haute doit decroitre quand le coup cesse d'etre bon : c'est ce
// que faisait age_history(), desormais porte par le terme de gravite.
TEST(HistoryGravity, DecaysWhenMoveStopsBeingGood)
{
    int e = SearchWorker::HistMax / 2;
    const int before = e;
    SearchWorker::update_hist(e, -100);
    EXPECT_LT(e, before);
}
