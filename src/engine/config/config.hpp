#pragma once

#ifdef SPSA_TUNING
#define PARAM_SPECIFIER inline
#else
#define PARAM_SPECIFIER constexpr
#endif

namespace engine_constants
{
    namespace eval
    {
        constexpr int MateScore = 10000;
        constexpr int TacticalScore = 8500; // Move scoring
        constexpr int Inf = 10000;
        constexpr int SyzygyScore = 9000;
        constexpr int SyzygyMaxPieces = 5;
    }
#ifdef NNUE_EVAL
    // Search parameters tuned for the NNUE evaluation
    namespace search
    {
        constexpr int MaxDepth = 64;

        namespace aspiration
        {
            PARAM_SPECIFIER int EnableDepth = 5;
            PARAM_SPECIFIER int MidDepth = 8;
            PARAM_SPECIFIER int HighDepth = 12;

            PARAM_SPECIFIER int SmallDelta = 16;
            PARAM_SPECIFIER int MidDelta = 41;
            PARAM_SPECIFIER int HighDelta = 20;

            PARAM_SPECIFIER int WidenMinDelta = 50;
            PARAM_SPECIFIER int WidenMaxDelta = 2000;
            PARAM_SPECIFIER int MaxIterations = 4;
            PARAM_SPECIFIER int MateWindowMargin = 256;
        }

        namespace razoring
        {
            PARAM_SPECIFIER int MaxDepth = 3;
            PARAM_SPECIFIER int MarginDepthFactor = 100;
            PARAM_SPECIFIER int MarginConst = 0;
        }
        namespace reverse_futility_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 4;
            PARAM_SPECIFIER int MarginDepthFactor = 73;
            PARAM_SPECIFIER int MarginConst = 61;
        }
        namespace iterative_deepening
        {
            PARAM_SPECIFIER int MaxDepth = 6;
            PARAM_SPECIFIER int NewDepthIncr = 4;
        }
        namespace null_move_pruning
        {
            PARAM_SPECIFIER int MinDepth = 2;
            PARAM_SPECIFIER int RConst = 4;
            PARAM_SPECIFIER int RDiv = 4;
        }
        namespace futility_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 8;
            PARAM_SPECIFIER int MarginConst = 95;
            PARAM_SPECIFIER int MarginDepthFactor = 105;
        }
        namespace singular
        {
            PARAM_SPECIFIER int MinDepth = 8;
        }
        namespace null_move_reduction
        {
            PARAM_SPECIFIER int MaxDepth = 4;
            PARAM_SPECIFIER int MaxMovesConst = 8;
            PARAM_SPECIFIER int MaxMovesDepthSqFactor = 2;
        }
        namespace late_move_reduction
        {
            PARAM_SPECIFIER int MinDepth = 3;
            PARAM_SPECIFIER int MinMovesSearched = 5;
            PARAM_SPECIFIER int MaxDepthReduction = 1;

            PARAM_SPECIFIER double TableInitConst = 0.63065940599962;
            PARAM_SPECIFIER double TableInitDiv = 2.301959991800665;
        }
        namespace time
        {
            // Gestion du temps. Le moteur n'avait qu'UNE limite, donc il
            // demarrait des iterations qu'il ne pouvait pas finir, les
            // avortait et jetait leur travail.
            //
            // BaseDivisor : part de l'horloge restante allouee a ce coup.
            // IncrementFraction : part de l'increment ajoutee (en %).
            // HardFactor : la limite dure vaut ce multiple de la souple --
            //   elle n'autorise que de FINIR une iteration deja commencee.
            // HardClampPercent : et jamais plus que ce % de l'horloge, pour
            //   ne pas jouer toute sa pendule sur un coup.
            // InstabilityFactor : si le meilleur coup racine vient de
            //   changer, la limite souple est etiree d'autant -- la position
            //   n'est pas tranchee, une iteration de plus vaut son prix.
            PARAM_SPECIFIER int BaseDivisor = 28;
            PARAM_SPECIFIER int IncrementFraction = 50;
            PARAM_SPECIFIER int HardFactor = 3;
            PARAM_SPECIFIER int HardClampPercent = 40;
            PARAM_SPECIFIER double InstabilityFactor = 1.5;
            // On ne teste la limite souple qu'APRES une iteration terminee,
            // donc demarrer l'iteration suivante double a peu pres le temps
            // total (EBF mesure ~2,2). Il faut donc renoncer bien avant la
            // limite : a EBF 2, une iteration coute autant que toutes les
            // precedentes cumulees, donc si on a deja depense plus de la
            // moitie du budget elle ne rentrera pas.
            // Mesure sans ce garde-fou : 0,8-1,1 s consommees pour une
            // limite souple de 403 ms.
            PARAM_SPECIFIER int SoftStartPercent = 50;
        }
        namespace see_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 6;
            PARAM_SPECIFIER int ThresholdDepthFactor = 15;
        }
    }
#else
    // Search parameters tuned for the HCE evaluation
    namespace search
    {
        constexpr int MaxDepth = 64;

        namespace aspiration
        {
            PARAM_SPECIFIER int EnableDepth = 5;
            PARAM_SPECIFIER int MidDepth = 8;
            PARAM_SPECIFIER int HighDepth = 12;

            PARAM_SPECIFIER int SmallDelta = 15;
            PARAM_SPECIFIER int MidDelta = 32;
            PARAM_SPECIFIER int HighDelta = 25;

            PARAM_SPECIFIER int WidenMinDelta = 50;
            PARAM_SPECIFIER int WidenMaxDelta = 2000;
            PARAM_SPECIFIER int MaxIterations = 5;
            PARAM_SPECIFIER int MateWindowMargin = 256;
        }

        namespace razoring
        {
            PARAM_SPECIFIER int MaxDepth = 3;
            PARAM_SPECIFIER int MarginDepthFactor = 100;
            PARAM_SPECIFIER int MarginConst = 0;
        }
        namespace reverse_futility_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 7;
            PARAM_SPECIFIER int MarginDepthFactor = 57;
            PARAM_SPECIFIER int MarginConst = 55;
        }
        namespace iterative_deepening
        {
            PARAM_SPECIFIER int MaxDepth = 6;
            PARAM_SPECIFIER int NewDepthIncr = 4;
        }
        namespace null_move_pruning
        {
            PARAM_SPECIFIER int MinDepth = 2;
            PARAM_SPECIFIER int RConst = 3;
            PARAM_SPECIFIER int RDiv = 5;
        }
        namespace futility_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 8;
            PARAM_SPECIFIER int MarginConst = 82;
            PARAM_SPECIFIER int MarginDepthFactor = 105;
        }
        namespace singular
        {
            PARAM_SPECIFIER int MinDepth = 8;
        }
        namespace null_move_reduction
        {
            PARAM_SPECIFIER int MaxDepth = 4;
            PARAM_SPECIFIER int MaxMovesConst = 8;
            PARAM_SPECIFIER int MaxMovesDepthSqFactor = 2;
        }
        namespace late_move_reduction
        {
            PARAM_SPECIFIER int MinDepth = 3;
            PARAM_SPECIFIER int MinMovesSearched = 5;
            PARAM_SPECIFIER int MaxDepthReduction = 2;

            PARAM_SPECIFIER double TableInitConst = 0.6295;
            PARAM_SPECIFIER double TableInitDiv = 2.3783;
        }
        namespace time
        {
            // Gestion du temps. Le moteur n'avait qu'UNE limite, donc il
            // demarrait des iterations qu'il ne pouvait pas finir, les
            // avortait et jetait leur travail.
            //
            // BaseDivisor : part de l'horloge restante allouee a ce coup.
            // IncrementFraction : part de l'increment ajoutee (en %).
            // HardFactor : la limite dure vaut ce multiple de la souple --
            //   elle n'autorise que de FINIR une iteration deja commencee.
            // HardClampPercent : et jamais plus que ce % de l'horloge, pour
            //   ne pas jouer toute sa pendule sur un coup.
            // InstabilityFactor : si le meilleur coup racine vient de
            //   changer, la limite souple est etiree d'autant -- la position
            //   n'est pas tranchee, une iteration de plus vaut son prix.
            PARAM_SPECIFIER int BaseDivisor = 28;
            PARAM_SPECIFIER int IncrementFraction = 50;
            PARAM_SPECIFIER int HardFactor = 3;
            PARAM_SPECIFIER int HardClampPercent = 40;
            PARAM_SPECIFIER double InstabilityFactor = 1.5;
            // On ne teste la limite souple qu'APRES une iteration terminee,
            // donc demarrer l'iteration suivante double a peu pres le temps
            // total (EBF mesure ~2,2). Il faut donc renoncer bien avant la
            // limite : a EBF 2, une iteration coute autant que toutes les
            // precedentes cumulees, donc si on a deja depense plus de la
            // moitie du budget elle ne rentrera pas.
            // Mesure sans ce garde-fou : 0,8-1,1 s consommees pour une
            // limite souple de 403 ms.
            PARAM_SPECIFIER int SoftStartPercent = 50;
        }
        namespace see_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 6;
            PARAM_SPECIFIER int ThresholdDepthFactor = 15;
        }
    }
#endif
}