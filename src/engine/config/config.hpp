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
            // Seconde passe du RFP, sur le reseau complet. On teste d'abord
            // la tete PSQT (~30x moins chere : 32 octets par ligne
            // d'accumulateur contre 1024) ; si elle n'elague pas, on refait
            // le MEME test avec le reseau complet, sous sa propre marge.
            //
            // PreciseMarginConst = MarginConst + 250, et ce +250 est MESURE,
            // pas choisi. Recopier MarginConst tel quel serait une faute :
            // sur la population des noeuds RFP, la tete PSQT lit ~250 a 330
            // centipions SOUS le reseau complet (mesure sur 1,6 M
            // echantillons, moyenne plate de +296/+328/+319/+338 aux
            // profondeurs 1 a 4). Le seuil que le SPSA a reellement valide
            // pour le test lazy est donc full >= beta + margin + 250 : la
            // marge tunee contient ce decalage. Sans la correction, le test
            // precis serait 250cp plus permissif que celui qui a ete valide,
            // et son gain apparent en noeuds ne serait qu'un elagage plus
            // agressif.
            //
            // Ce decalage n'existe pas aux positions racines, ou l'ecart
            // vaut 0 a 89cp. Il est propre aux noeuds RFP : non-PV, faible
            // profondeur, atteints apres que l'ordonnancement a pousse
            // captures et killers en premier -- des positions tendues, ou le
            // trunk (qui porte les features de menaces) diverge legitimement
            // d'un terme materiel-et-placement.
            //
            // Point de depart calibre, et pas sur une borne, donc le SPSA
            // peut perturber des deux cotes (contrairement a 0).
            //
            // Parametres INDEPENDANTS, pas un delta ajoute a la marge lazy :
            // un facteur a tuner ne doit pas dependre d'un autre facteur a
            // tuner, sinon les deux dimensions se masquent mutuellement.
            //
            // Mettre PreciseMarginDepthFactor tres haut desactive la seconde
            // passe et redonne le comportement d'avant.
            PARAM_SPECIFIER int PreciseMarginDepthFactor = 73;
            PARAM_SPECIFIER int PreciseMarginConst = 311;
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
            // StabilityFactor0..4 : la limite souple est etiree ou raccourcie
            //   selon la DUREE de stabilite du meilleur coup racine (voir
            //   plus bas, chiffres a l'appui).
            PARAM_SPECIFIER int BaseDivisor = 28;
            PARAM_SPECIFIER int IncrementFraction = 50;
            PARAM_SPECIFIER int HardFactor = 3;
            PARAM_SPECIFIER int HardClampPercent = 40;
            // Facteur applique a la limite souple selon la DUREE de stabilite
            // du meilleur coup racine, indexe par le nombre d'iterations
            // consecutives sans changement (>=4 -> derniere case).
            //
            // Mesure sur 300 positions / 2398 iterations, probabilite que le
            // coup change a l'iteration SUIVANTE :
            //   stable depuis 0 iter : 31,1 %   (n=469)
            //   stable depuis 1 iter : 28,4 %   (n=394)
            //   stable depuis 2 iter : 19,5 %   (n=302)
            //   stable depuis 3 iter : 16,3 %   (n=313)
            //   stable depuis 4+ iter:  9,8 %   (n=245)
            // Taux de base : 18,1 %. Soit un ecart de 3x, monotone -- la
            // DUREE de stabilite predit bien mieux que le simple booleen
            // "ca vient de changer", qui melangeait les cases 0 et 1 avec
            // tout le reste. Les facteurs suivent approximativement le
            // rapport de chaque case au taux de base.
            PARAM_SPECIFIER double StabilityFactor0 = 1.50;
            PARAM_SPECIFIER double StabilityFactor1 = 1.35;
            PARAM_SPECIFIER double StabilityFactor2 = 1.05;
            PARAM_SPECIFIER double StabilityFactor3 = 0.95;
            PARAM_SPECIFIER double StabilityFactor4 = 0.75;
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
            // StabilityFactor0..4 : la limite souple est etiree ou raccourcie
            //   selon la DUREE de stabilite du meilleur coup racine (voir
            //   plus bas, chiffres a l'appui).
            PARAM_SPECIFIER int BaseDivisor = 28;
            PARAM_SPECIFIER int IncrementFraction = 50;
            PARAM_SPECIFIER int HardFactor = 3;
            PARAM_SPECIFIER int HardClampPercent = 40;
            // Facteur applique a la limite souple selon la DUREE de stabilite
            // du meilleur coup racine, indexe par le nombre d'iterations
            // consecutives sans changement (>=4 -> derniere case).
            //
            // Mesure sur 300 positions / 2398 iterations, probabilite que le
            // coup change a l'iteration SUIVANTE :
            //   stable depuis 0 iter : 31,1 %   (n=469)
            //   stable depuis 1 iter : 28,4 %   (n=394)
            //   stable depuis 2 iter : 19,5 %   (n=302)
            //   stable depuis 3 iter : 16,3 %   (n=313)
            //   stable depuis 4+ iter:  9,8 %   (n=245)
            // Taux de base : 18,1 %. Soit un ecart de 3x, monotone -- la
            // DUREE de stabilite predit bien mieux que le simple booleen
            // "ca vient de changer", qui melangeait les cases 0 et 1 avec
            // tout le reste. Les facteurs suivent approximativement le
            // rapport de chaque case au taux de base.
            PARAM_SPECIFIER double StabilityFactor0 = 1.50;
            PARAM_SPECIFIER double StabilityFactor1 = 1.35;
            PARAM_SPECIFIER double StabilityFactor2 = 1.05;
            PARAM_SPECIFIER double StabilityFactor3 = 0.95;
            PARAM_SPECIFIER double StabilityFactor4 = 0.75;
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