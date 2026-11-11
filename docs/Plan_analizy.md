# Plan analizy

Wersja robocza z 11.11.2026, do zamrożenia przed uruchomieniem przebiegu głównego.

Plan ustala z góry, które porównania są wnioskami głównymi, a które analizami dodatkowymi. Zamrożenie polega na zapisaniu go w repozytorium (commit z datą) przed policzeniem prognoz testowych. Każda późniejsza zmiana trafia do sekcji 7 z datą i uzasadnieniem, a w artykule opisujemy ją jako odstępstwo od planu.

## 1. Pytanie badawcze

Czy w warunkach operacyjnych (prognoza wydawana o północy na 48 godzin, codzienna estymacja na oknie 84 dób, wyłącznie informacja dostępna w chwili prognozy) łączenie prognoz pięciu modeli z pierwotnej wersji artykułu daje dokładniejszą prognozę na dobę D+1 niż modele pojedyncze, i który sposób łączenia jest najlepszy?

Odniesieniem dla sposobów łączenia jest średnia arytmetyczna (C3). W literaturze często wygrywa ona z wagami estymowanymi („zagadka łączenia prognoz”: Stock i Watson 2004; Smith i Wallis 2009; Claeskens i in. 2016), więc to ona jest poprzeczką, którą metoda z wagami musi przeskoczyć.

## 2. Dane, okresy, prognozy

- **Zbiór główny:** ElecEner Long z korektami z `configs/datasets/elecener.yaml`.
- **Identyfikacja** (struktury i hiperparametry wszystkich modeli, także benchmarku LightGBM): 1.11.2001-31.03.2002.
- **Rozgrzewka łączenia:** od 1.07.2002.
- **Test:** 1.08.2002-29.06.2003 (333 doby). Data startu testu zależy wyłącznie od odpowiedzi właściciela danych w sprawie kwietnia 2002, nie od wyników:
  - kwiecień potwierdzony jako poprawny: test od 1.05.2002 (425 dób); prognozy dla dodatkowych dób dopisujemy do tego samego archiwum;
  - kwiecień uznany za błędny: test od 22.08.2002 (312 dób), żeby kwiecień nie wchodził do okien uczących prognoz używanych jako historia wag;
  - brak odpowiedzi przed oceną: wariant 1.08.2002, z opisem wpływu kwietnia na wagi z początku sierpnia jako ograniczeniem.
- **Pula główna:** MR_ARIMA, ARIMA, ES_ARIMA, SVR_ARIMA, NN_ARIMA.
- **Benchmarki:** SNAIVE, HONG, LGBM.
- **Pule rozszerzone** (analiza dodatkowa): `plus_hong`, `plus_lgbm`.
- **Sposoby łączenia:**
  - C0: wagi z wartości dopasowanych, procedura artykułu, 48 h; warianty C0_H72 i C0_H168;
  - C1: regresja na prognozach;
  - C2: wagi nieujemne sumujące się do 1;
  - C3: średnia;
  - C4: mediana;
  - C5: wagi odwrotne do MSE;
  - C6: wybór modelu najlepszego w ostatnich K dobach.
  - Dla C1, C2, C5 i C6: K ∈ {2, 3, 7, 14, 28}.
- **Scenariusze temperatury:**
  - temperatura zmierzona (oracle);
  - persystencja;
  - siatka błędu losowego: odchylenie dla D+1 od 0,5 do 3,5 °C co 0,5 °C, dla D 0,7 tej wartości, 10 losowań na poziom, wspólne liczby losowe.

## 3. Analiza główna (potwierdzająca)

- **Miara:** MAE doby D+1 (godziny 25-48), liczone na parach (doba wydania, blok) wspólnych dla wszystkich porównywanych wariantów.
- **Pula:** główna.
- **Scenariusze:** temperatura zmierzona i persystencja, oba raportowane jako główne. Wniosek uznajemy za potwierdzony, gdy kierunek różnicy jest ten sam w obu scenariuszach. Gdy kierunki się różnią, wynik opisujemy jako zależny od jakości informacji o temperaturze.
- **Test:** Diebolda-Mariano na dziennych stratach (średnie MAE doby D+1), wariancja Neweya-Westa, poprawka Harveya, Leybourne'a i Newbolda, test dwustronny, α = 0,05 po poprawce Holma w obrębie rodziny.
- **Rodziny hipotez** (każde porównanie: wariant kontra C3):

| Rodzina | Pytanie | Porównania |
|---|---|---|
| H1 | Czy wagi estymowane na prognozach wygrywają ze średnią? | C1, C2, C5 × 5 wartości K (15) |
| H2 | Czy procedura pierwotna (C0, 48 h) jest gorsza od średniej? | C0 (1) |
| H3 | Czy średnia wygrywa z modelami pojedynczymi i benchmarkami? | 5 modeli bazowych + SNAIVE, HONG, LGBM (8) |
| H4 | Czy wybór modelu wygrywa z łączeniem? | C6 × 5 wartości K (5) |

- **Zbiór modeli najlepszych (MCS):** metoda T_max, α = 0,10, na modelach pojedynczych i wariantach łączenia puli głównej.
- **Przedziały ufności różnic względem C3:** bootstrap blokowy (bloki 7 dób, 2000 powtórzeń).
- **Interpretacja K:** K nie jest parametrem wybieranym na teście. Raportujemy wszystkie wartości i opisujemy przebieg błędu w funkcji K: krótka historia daje szum w wagach, długa daje wagi nieaktualne.

## 4. Analizy dodatkowe (opisowe, bez wniosków potwierdzających)

1. **Pojedyncze okno 48 h:** odsetek dób, w których łączenie miałoby mniejsze MAE z 48 godzin niż najlepszy model bazowy wybrany po fakcie (układ oceny z pierwotnej wersji artykułu). Pokazuje, jak niepewny był wniosek z jednego okna.
2. **Różnorodność błędów:** rozkład MSE średniej na średni MSE modeli i różnorodność (Krogh i Vedelsby 1995), średnia korelacja błędów; dla puli głównej i pul rozszerzonych.
3. **Pule rozszerzone** `plus_hong` i `plus_lgbm`: te same miary i testy, opisowo.
4. **Warianty C0 z 72 i 168 godzinami:** czy słabość C0 wynika z idei wartości dopasowanych, czy z krótkiego okna.
5. **Wrażliwość na błąd prognozy temperatury:** krzywa MAE doby D+1 na siatce poziomów i próg opłacalności, czyli poziom błędu, przy którym model z temperaturą zrównuje się z najlepszym modelem bez temperatury. Punkt odniesienia „prognoza z około 2002 roku” wyznaczamy z raportów weryfikacji ECMWF (Tech. Memo 414 i 432) przed oceną wyników.
6. **Korekta ARIMA:** model dwustopniowy a jego pierwszy stopień (DM). Pierwsze stopnie (regresja, SVR, sieć, wygładzanie) zestawiamy też z LightGBM: wszystkie bez korekty ARIMA, strojone jako samodzielne modele (sekcja 2), na tych samych dobach. Rozdziela to przewagę wynikającą z modelu od przewagi z korekty ARIMA; LightGBM nie dostaje korekty ARIMA, bo jego reszty w oknie uczącym są prawie zerowe (dopasowanie drzew) i nie odpowiadają błędom prognozy.
7. **Przekroje:** doba D (godziny 1-24); RMSE i MASE; typy dni (robocze, soboty, niedziele, święta, dni pomostowe); miesiące; godziny horyzontu; bez godzin oznaczonych jako podejrzane.
8. **Stabilność wag:** zmienność z doby na dobę, odsetek wag ujemnych, częstość zmiany modelu w C6.
9. **Koszt obliczeń:** czas dopasowania i prognozy modeli.
10. **Replikacja na GEFCom2012:** ten sam plan, porównanie kierunków wyników H1-H4 z ElecEner.
11. **Kontrola metody estymacji SARIMA:** przebieg kontrolny z dokładną wiarygodnością na podzbiorze dób.

## 5. Zasady stałe

- **Bez wyborów na teście:** żaden wybór (K, warianty modeli, rzędy ARIMA, hiperparametry, data startu testu) nie jest dokonywany na podstawie wyników okresu testowego.
- **Awaria modelu w dobie:** brak prognozy łączonej w tej dobie. Wagi w kolejnych dobach liczone na dostępnej historii, jeśli brakuje najwyżej jednej doby na tydzień. Liczba awarii raportowana.
- **Porównania:** wyłącznie na wspólnych parach (doba wydania, blok).
- **Scenariusz losowy:** miary uśrednione po losowaniach; testy na dziennych stratach uśrednionych po losowaniach.
- **Powtarzalność:** ziarna stałe, migawka wejść w katalogu przebiegu, ślad każdego wywołania w `runs/`.

## 6. Do rozstrzygnięcia przed oceną (nie przed obliczeniami)

- **Data startu testu** (sekcja 2), zależnie od odpowiedzi w sprawie kwietnia 2002.
- **Pasek „prognoza z około 2002 roku”** na siatce błędu temperatury (sekcja 4, punkt 5).

## 7. Odstępstwa od planu

(puste)
