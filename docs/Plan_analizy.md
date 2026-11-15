# Plan analizy

Wersja 2 z 13.11.2026 (po uwagach audytora i Profesora), zamrożona 13.11.2026 przed obliczeniem prognoz testowych (etykieta `plan-v2`).

Plan ustala z góry, które porównania są wnioskami głównymi, a które analizami dodatkowymi. Zamrożenie polega na oznaczeniu stanu repozytorium etykietą z datą przed obliczeniem prognoz testowych. Rodziny hipotez są zapisane także w konfiguracji eksperymentu (`evaluation.families`), więc raport liczy dokładnie to, co tu ustalono. Każda późniejsza zmiana trafia do sekcji 9 z datą i uzasadnieniem, a w artykule opisujemy ją jako odstępstwo od planu.

## 1. Pytanie badawcze

Czy w warunkach operacyjnych (prognoza wydawana o północy na 48 godzin, codzienna estymacja na oknie 84 dób, wyłącznie informacja dostępna w chwili prognozy) regresja łącząca prognozy pięciu modeli z pierwotnej wersji artykułu daje dokładniejszą prognozę na dobę D+1 niż modele pojedyncze i niż zwykła średnia tych prognoz?

Metodą artykułu jest regresja łącząca: współczynniki regresji na prognozach z ostatnich 2 dób są wagami modeli (C1, K = 2), kontrolnie z 3 i 7 dób. Wersja pierwotna liczyła tę regresję na wartościach dopasowanych (C0). Średnia arytmetyczna (C3) jest punktem odniesienia, którego wymagają recenzenci; w literaturze często wygrywa ona z wagami estymowanymi („zagadka łączenia prognoz”: Stock i Watson 2004; Smith i Wallis 2009; Claeskens i in. 2016). Teza artykułu zostanie sformułowana po wynikach, ale hipotezy i testy są ustalone tutaj.

Wydanie o północy jest konwencją symulacji. W praktyce opisanej w artykule dane obejmują okres do północy, praca zaczyna się o 6:00, a zamówienie składa się do 11:00.

## 2. Dane, okresy, prognozy

- **Zbiór główny:** ElecEner Long z korektami z `configs/datasets/elecener.yaml`. Plik Short dotyczy innego obszaru; służy wyłącznie do rekonstrukcji wyniku pierwszej wersji (sekcja 4, punkt 14).
- **Identyfikacja** (struktury i hiperparametry wszystkich modeli): 1.11.2001-31.03.2002. Doby walidacyjne rozłożone równomiernie na dostępną część okresu (od pierwszej doby z pełnym oknem), z krokiem niebędącym wielokrotnością tygodnia; ich daty zapisuje wynik identyfikacji.
- **Strojenie** SVR, sieci i LightGBM: Optuna (100 prób) na pierwszym stopniu przy temperaturze zmierzonej, potem sito: 5 najlepszych kandydatów sprawdzanych jako pełny model (z korektą ARIMA reszt; LightGBM sam) przy temperaturze z persystencji, na tych samych dobach walidacyjnych.
- **Rozgrzewka łączenia:** od 1.07.2002. **Test:** 1.08.2002-29.06.2003 (333 doby).
- **Wpływ kwietnia 2002 na wagi** (uwaga audytora): przy K ≤ 7 historia wag dla wydania z 1.08.2002 sięga wydań, których okna uczące zaczynają się najwcześniej 1.05.2002, więc analiza główna (K = 2, 3, 7) nie widzi kwietnia. Dla wariantów dodatkowych z K = 14 i K = 28 kwiecień wchodzi do okien wydań używanych jako historia wag do 7.08 i 21.08.2002.
- **Reguła daty startu** (zależy wyłącznie od odpowiedzi właściciela danych, nie od wyników):
  - kwiecień potwierdzony: test od 1.05.2002 (425 dób), rozgrzewka dopisana od 2.04.2002; prognozy dopisujemy do tego samego archiwum;
  - kwiecień błędny albo brak odpowiedzi: analiza główna od 1.08.2002; warianty z K = 14 od 8.08.2002, z K = 28 od 22.08.2002. W kodzie reguła ogólna (`evaluation.excluded_data`): para (wariant, doba wydania, blok) wchodzi do oceny tylko wtedy, gdy żadne okno uczące jej prognozy, łącznie z oknami prognoz z historii wag, nie obejmuje kwietnia 2002; daty 1.08, 8.08 i 22.08 wynikają z niej same (test w kodzie).
- **Pula główna:** MR_ARIMA, ARIMA, ES_ARIMA, SVR_ARIMA, NN_ARIMA. **Benchmarki:** SNAIVE, HONG, LGBM.
- **Modele pomocnicze:** HONG_ARIMA (HONG z tą samą korektą reszt co MR_ARIMA, do puli `plus_hong`), ARIMA_S24 (okres sezonowy 24 h, ablacja dla R1).
- **Pule rozszerzone** (opisowo): `plus_hong` (pula główna + HONG_ARIMA), `plus_lgbm` (pula główna + LGBM).
- **Sposoby łączenia** (liczone wszystkie; w analizie głównej tylko C0, C1 i C3):
  - C0: regresja na wartościach dopasowanych z 48 h okna (procedura pierwotna); warianty C0_H72, C0_H168;
  - C1: regresja na prognozach tego samego bloku z ostatnich K dób;
  - C2: wagi nieujemne sumujące się do 1; C3: średnia; C4: mediana; C5: wagi odwrotne do MSE;
  - C6: wybór modelu o najmniejszym MSE w ostatnich K dobach (MSE, bo wszystkie metody z estymowanymi wagami minimalizują kwadraty błędów, a C6 jest granicą C5);
  - C7: średnia z korektą poziomu (C3 + b); C8: a + b · C3 (poziom i wspólna skala przy równych udziałach modeli);
  - K ∈ {2, 3, 7, 14, 28} dla C1, C2, C5-C8; analiza główna: K ∈ {2, 3, 7}.
- **Scenariusze temperatury:** zmierzona (oracle), persystencja, siatka błędu losowego (odchylenie dla D+1 od 0,5 do 3,5 °C co 0,5 °C, dla D 0,7 tej wartości, 10 losowań na poziom, wspólne liczby losowe).

## 3. Analiza główna (potwierdzająca)

- **Miara:** MAE doby D+1 (godziny 25-48).
- **Zbiór dób:** pary (doba wydania, blok) wspólne dla wariantów występujących w rodzinach hipotez. Brak wariantu dodatkowego nie wycina dób z analizy głównej. Zestawienie wszystkich wariantów (opisowe) liczymy na tych samych dobach; warianty bez kompletu tych dób (krótsza historia, reguła kwietnia) na swoich dobach, z liczbą dób w tabeli, i nie porównujemy ich w progu opłacalności. Profil K liczymy na dobach wspólnych dla wszystkich K. Tabela pokrycia podaje dla każdego planowanego wariantu doby planowane, dostępne i użyte.
- **Rodziny hipotez** (różnica = wariant - odniesienie rodziny; ujemna: wariant lepszy):

| Rodzina | Pytanie | Odniesienie | Porównania |
|---|---|---|---|
| H1 | Czy regresja łącząca wygrywa ze zwykłą średnią? | C3 | C1_K2, C1_K3, C1_K7 (3) |
| H2 | Czy wagi z wartości dopasowanych są gorsze od wag z prognoz, na tych samych 48 obserwacjach (doby D-2 i D-1)? | C1_K2 | C0 (1) |
| H3 | Czy regresja łącząca wygrywa z każdym modelem pojedynczym i benchmarkiem? | C1_K2 | MR_ARIMA, ARIMA, ES_ARIMA, SVR_ARIMA, NN_ARIMA, SNAIVE, HONG, LGBM (8) |

- **Test:** Diebolda-Mariano na dziennych stratach (MAE bloku D+1 dla każdej doby wydania; w scenariuszu losowym średnia po losowaniach), wariancja Neweya-Westa z ⌊n^(1/3)⌋ opóźnieniami liczona według rzeczywistych odstępów dat (luki nie sklejają dób), poprawka Harveya, Leybourne'a i Newbolda z h = 1, rozkład t z n - 1 stopniami swobody, test dwustronny.
- **Kontrole konwencji testu** (raportowane obok, bez zmiany reguły): h = 2 w poprawce HLN, pasmo 14 dób, wariancja HAC bez HLN.
- **Poprawka na wielokrotność:** Holm w obrębie rodziny, z planowaną liczebnością rodziny. Kontrast, którego nie da się ocenić, zostaje w tabeli jako nieoceniony i liczy się jak p = 1.
- **Reguła potwierdzenia w dwóch scenariuszach:** dla każdego kontrastu bierzemy większą z dwóch dwustronnych wartości p (temperatura zmierzona, persystencja), stosujemy Holma w rodzinie (α = 0,05) i wymagamy zgodnego kierunku różnicy w obu scenariuszach. Efekty i przedziały podajemy dla obu scenariuszy. Kontrast potwierdzony w jednym scenariuszu opisujemy jako zależny od jakości informacji o temperaturze.
- **Zbiór modeli najlepszych (MCS):** T_max, α = 0,10, na wariantach z rodzin hipotez; interpretujemy go jako zbiór wariantów, których procedura nie wyeliminowała.
- **Przedziały ufności różnic względem C3:** bootstrap blokowy po dobach kalendarza (bloki 7 dób, 2000 powtórzeń; doby wykluczone zostają w kalendarzu jako puste).
- **Interpretacja K:** K nie jest wybierane na teście; K = 2 to wariant artykułu.

## 4. Analizy dodatkowe (opisowe, bez wniosków potwierdzających)

Tekst główny:
1. **Pojedyncze okno 48 h:** odsetek dób, w których łączenie miałoby mniejsze MAE z 48 h niż najlepszy model bazowy wybrany po fakcie (układ oceny pierwszej wersji).
2. **Rozkład MAE średniej:** dla każdej godziny średnie |e| modeli - |błąd średniej| = 2 min(P, N), gdzie P i N to średnie dodatnich i ujemnych części błędów; uśrednione po godzinach doby D+1, także według godziny doby; porównanie z różnicą między średnim MAE modeli a MAE każdego modelu pojedynczego.
3. **Profil błędu w funkcji K** dla C1, C2, C5-C8 wobec C3.

Analizy dodatkowe:

4. **Pozostałe sposoby łączenia:** C2, C4, C5, C6 (wybór zamiast łączenia), C7 i C8 (kalibrowana średnia: czy zysk C1 wynika z kalibracji średniej, czy z różnicowania wag), K = 14 i 28.
5. **Wartości dopasowane kontra prognozy przy tej samej liczbie obserwacji:** C0_H72 kontra C1_K3 i C0_H168 kontra C1_K7, tylko na wydaniach z pełną historią K dób (para 48 h / K = 2 jest w H2).
6. **Pule rozszerzone** `plus_hong` i `plus_lgbm`.
7. **Korekta ARIMA:** model dwustopniowy kontra jego pierwszy stopień; pierwsze stopnie (MR, ES, SVR, NN) kontra LGBM, wszystkie bez korekty ARIMA.
8. **ARIMA_S24:** okres sezonowy 24 h kontra 168 h (R1).
9. **Wrażliwość na błąd prognozy temperatury:** krzywa MAE doby D+1 na siatce i próg opłacalności wobec najlepszego modelu bez temperatury (modele bez temperatury wyznaczone z ich wejść; odniesienie wybrane opisowo na teście; liczba przecięć i niepewność raportowane). Raporty weryfikacji ECMWF (Tech. Memo 414 i 432) wyłącznie jako kontekst.
10. **Przekroje:** doba D; RMSE (w scenariuszu losowym: pierwiastek ze średniej MSE po godzinach i losowaniach) i MASE; typy dni z liczebnościami (święta, dni pomostowe); miesiące; godziny horyzontu; bez godzin oznaczonych jako podejrzane.
11. **Koszt obliczeń:** osobno dopasowanie, jedna prognoza 48 h (poranne uruchomienie operacyjne) i pętla po scenariuszach (koszt badania), z procesorem i liczbą wątków.
12. **Replikacja na GEFCom2012:** ten sam plan; porównujemy kierunki wyników H1-H3 z ElecEner. Przy GEFCom błąd temperatury stały w dobie bada błąd poziomu przy znanym profilu godzinowym.
13. **Kontrola estymacji SARIMA:** dokładna wiarygodność zamiast CSS dla wszystkich członów ARIMA na ciągłym fragmencie: wydania 1-31.10.2002 z rozgrzewką od 22.09.2002 (historia wag dla K ≤ 7); porównanie rankingu wariantów z rodzin hipotez. Kody zbieżności raportowane osobno od awarii.
14. **Rekonstrukcja wyniku pierwszej wersji (plik Short):** odtworzenie tabeli 1 i wyników MAPE z podziałem na doby D i D+1; na kilku dobach, które Short pozwala ocenić, C1 i C3 wyłącznie jako ilustracja, bez testów.

Suplement:

15. Rozkład błędu kwadratowego średniej (Krogh i Vedelsby 1995) i korelacje błędów pul.
16. Stabilność wag (zmienność z doby na dobę, odsetek wag ujemnych, częstość zmiany modelu w C6) oraz usunięcie jednej doby z historii wag C1 (K = 2 i 7): suma modułów zmian wag i średnia godzinowa zmiana prognozy D+1; mediana i 90. percentyl po wydaniach.
17. Koszt asymetryczny: niedoszacowanie U i przeszacowanie O (MAE = U + O), strata r U + O dla r ∈ {0,5; 1; 2; 4} i stosunek r*, przy którym warianty zamieniają kolejność; bez przeliczania na pieniądze.

Przebiegi dodatkowe (po przebiegu głównym; punkt 18 przesądzony, punkt 19 jeśli starczy czasu obliczeń):

18. **Kwartalna ponowna identyfikacja** (przyjęta 13.11.2026). Pytanie: ile kosztuje to, że hiperparametry i rzędy dobrano wyłącznie na miesiącach zimowych (R3: adaptacja). Identyfikacja przed 1.11.2002, 1.02.2003 i 1.05.2003, każda na poprzedzającym okresie tej samej długości co identyfikacja główna (151 dób), tą samą procedurą: te same kroki, liczba prób, sito i liczba dób walidacyjnych. Sierpień-październik na parametrach z identyfikacji głównej, bo identyfikacja przed 1.08.2002 obejmowałaby kwiecień 2002. Jedyne porównanie: C1_K2 oraz każdy model pojedynczy, wariant z parametrami stałymi (przebieg główny) kontra odświeżanymi, na wspólnych dobach wydania 1.11.2002-29.06.2003, w scenariuszach oracle i persystencja; różnica MAE D+1 z przedziałem bootstrap i DM bez korekty Holma, wyłącznie opisowo. Tylko ElecEner.
19. **Okno uczące 56 dób**, te same parametry z identyfikacji (zamrożone; ablacja lekko faworyzuje okno 84); tylko temperatura zmierzona i persystencja.

Odłożone (archiwum wystarcza, liczymy na prośbę recenzenta): rozkład różnicy oracle-persystencja na wkład bieżącej prognozy i historii wag.

## 5. Zasady stałe

- **Bez wyborów na teście:** żaden wybór (K, warianty modeli, rzędy ARIMA, hiperparametry, data startu testu) nie zależy od wyników okresu testowego.
- **Awaria modelu w dobie:** brak prognozy łączonej w tej dobie; wagi w kolejnych dobach na dostępnej historii, jeśli brakuje najwyżej jednej doby na tydzień; liczba awarii raportowana.
- **Parametry:** przebieg odmawia pracy na parametrach wstępnych (bez kompletnej identyfikacji) i na wyniku identyfikacji z innych danych; identyfikacja musi kończyć się przed pierwszą prognozą.
- **Powtarzalność:** ziarna stałe, migawka wejść w katalogu przebiegu, ślad każdego wywołania w `runs/`, wersja kodu w wyniku identyfikacji.

## 6. Kolejność obliczeń i dostęp do wyników

1. **Przebieg próbny na rozgrzewce** (test próbny 10-31.07.2002, poza okresem testu): tu oglądamy wszystkie tabele i wykresy, czasy i pamięć. Wykonany 13.11.2026 (31 dób, bez awarii, około 25 minut); poprawki przed zamrożeniem: reguła kwietnia w ocenie, zbiór dób zestawienia opisowego, próg opłacalności tylko na pełnym zbiorze dób, opis kodów zbieżności w raporcie stanu.
2. **Zamrożenie planu:** 13.11.2026, etykieta `plan-v2` w repozytorium (zatwierdził Grzegorz Migut). Od tej chwili każda zmiana planu, kodu analizy lub konfiguracji wpływająca na wyniki trafia do sekcji 9.
3. **Test kwartałami:** sierpień-październik, listopad-styczeń, luty-kwiecień, maj-czerwiec. Między kwartałami wyłącznie kontrola techniczna (`prognozowanie status`: kompletność, awarie, zbieżność, czasy, rozmiar archiwum), bez miar dokładności.
4. **Ocena dokładności** raz, po ostatnim kwartale, zgodnie z tym planem.
5. **Błąd w kodzie wykryty w trakcie:** poprawka, przeliczenie dotkniętych modeli od początku testu i wpis w sekcji 9.
6. **Po przebiegu głównym:** GEFCom2012, potem przebieg z kwartalną ponowną identyfikacją (punkt 18), potem pozostałe przebiegi dodatkowe.

## 7. Ujawnienie wcześniejszego wglądu w doby testowe

Przed zamrożeniem planu część dób testowych była oglądana:
- kontrola wygładzania wykładniczego na 15 losowo wybranych dobach testu (daty nie zostały zapisane); wpłynęła na decyzję o wyborze wariantu wygładzania w identyfikacji (decyzja 12);
- porównanie CSS z dokładną wiarygodnością na 20 dobach: 2002-08-20, 08-21, 08-25, 09-17, 10-09, 10-28, 10-29, 11-08, 11-19, 12-03, 2003-01-03, 01-16, 02-15, 03-10, 04-13, 04-26, 05-05, 05-21, 06-06, 06-09 (decyzja 9);
- przebiegi techniczne kodu na parametrach wstępnych, które objęły doby wydania 1-6.08.2002 (także przebieg audytora 26.07-4.08.2002);
- kontrola optymalizatora CSS (13.11.2026) na oknie uczącym kończącym się 14.10.2002: porównane wyłącznie wartości funkcji celu, parametry i prognozy dwóch optymalizatorów między sobą, bez porównania z obserwacjami.

Analiza wrażliwości: kontrasty H1-H3 bez dób 1-6.08.2002 i bez 20 dób porównania CSS z dokładną wiarygodnością. Dób kontroli wygładzania nie da się wykluczyć, bo ich daty nie zostały zapisane.

## 8. Do rozstrzygnięcia przed oceną (nie przed obliczeniami)

- **Data startu testu** (sekcja 2), zależnie od odpowiedzi w sprawie kwietnia 2002.
- **Pasek „prognoza z około 2002 roku”** na siatce błędu temperatury (sekcja 4, punkt 9).

## 9. Odstępstwa od planu

Zmiany po zamrożeniu (etykieta `plan-v2`), z datą i uzasadnieniem.

| Data | Zmiana | Wpływ na wyniki |
|---|---|---|
| 13.11.2026 | Skrypt kolejki: Ctrl+C przerywa także uruchomione zadanie (`bcf5766`). | Brak: infrastruktura, kod analizy bez zmian. |
| 13.11.2026 | Ocena: reguła wyłączonych okresów (sekcja 1) była wywoływana przed dołączeniem kolumny bloku horyzontu, przez co pierwsza ocena przebiegu głównego przerwała się błędem, zanim zapisała jakikolwiek wynik. Poprawka kolejności i test regresji; definicja reguły bez zmian. | Brak: żadna tabela ani miara nie powstała przed poprawką. |
| 13.11.2026 | Jedno wydanie (1.11.2002) z niedomkniętą optymalizacją pierwszego stopnia ES_ARIMA (kod 1); prognoza powstała i zostaje w ocenie, jak każda inna. | Zapis informacyjny, bez przeliczeń. |
| 13.11.2026 | Konfiguracje przebiegów zaplanowanych w sekcji 6: GEFCom2012 z wynikiem swojej identyfikacji, przebieg kwartalny (`main_kwartalny.yaml`, punkt 18); strażnik przebiegu sprawdza kompletność każdego pliku z harmonogramu identyfikacji. | Brak na przebieg główny (kod i konfiguracja `main.yaml` bez zmian). |
| 14.11.2026 | Przebieg kwartalny (punkt 18): z parametrami z identyfikacji przed 1.02.2003 (SVR, jądro wielomianowe, C = 0,82) solver pierwszego stopnia SVR_ARIMA osiągnął limit iteracji (2 mln) w 88 z 89 wydań 1.02-30.04.2003; prognozy powstały. Bez przeliczeń (limit iteracji jest częścią procedury); w opisie punktu 18 wynik SVR_ARIMA dla tego kwartału z zastrzeżeniem. | Tylko analiza dodatkowa 18. |
| 14.11.2026 | Konfiguracja kontroli estymacji z sekcji 4 (`main_ml_kontrola.yaml`): modele i parametry jak w `main.yaml`, wszystkie człony SARIMA metodą dokładnej wiarygodności, wydania 1-31.10.2002, rozgrzewka od 22.09.2002 (test sprawdza, że różni się tylko metoda). | Brak na przebieg główny. |
| 14.11.2026 | Kontrola estymacji przerwana po 11 z 40 dób (kolejka przestała działać około 14:25, prawdopodobnie brak pamięci przy dwóch modelach SARIMA ML naraz); wznowienie z jednym dniem naraz (`execution.n_jobs: 1`). | Brak: ustawienie wykonania, policzone doby zostają. |

## 10. Analizy dodane po wynikach

Analizy w tej sekcji powstały po obejrzeniu wyników przebiegu głównego (ocena 13.11.2026). Każda jest opisana tu przed obliczeniem, z datą, i oznaczona w artykule jako dodana po wynikach. Nie zmieniają rodzin hipotez H1–H3 ani ich wyników.

### 10.1 Regresja łącząca ze ściąganiem wag do równych

Projekt z 15.11.2026, do zatwierdzenia przed obliczeniem.

**Powód.** Regresja łącząca (C1) przegrała ze średnią (C3) przy każdym K. Przewidywane pytanie recenzenta: czy wystarczyłoby ograniczyć szum wag, ściągając je do wag równych.

**Metoda.** Wariant C9 łączy archiwalne prognozy C1 i C3 tej samej doby, bloku, K, scenariusza i losowania: prognoza C9 = ψ · prognoza C1 + (1 − ψ) · prognoza C3. Jest to dokładnie regresja z wagami w = ψ · w_C1 + (1 − ψ) / n i wyrazem wolnym ψ · a_C1 (Diebold i Pauly 1990; Stock i Watson 2004). Przy ψ = 1 to C1, przy ψ = 0 to C3. Nic nie jest szacowane od nowa i nie powstaje żadna nowa prognoza modelu.

**Wartości ψ, ustalone z góry, wszystkie raportowane:**

1. Reguła Stocka i Watsona: ψ = max(0, 1 − κ · n / (T − 1 − n)), n = 5 modeli, κ ∈ {0,25; 0,5; 1}, w dwóch wersjach liczby obserwacji: (a) T = 24 · K godzin, jak w dosłownym zastosowaniu reguły; (b) T = K dób, bo błędy godzin tej samej doby są silnie skorelowane i doba jest bliższa niezależnej obserwacji. Wersja (a) daje ψ od 0,88 do 1,00, czyli prawie bez ściągania; wersja (b) daje ψ = 0 dla K ≤ 7 i od 0,38 do 0,94 dla K = 14 i 28. Reguła zależy tylko od K i n, nie od danych testu.
2. Stała siatka: ψ ∈ {0,25; 0,5; 0,75}.

**Zakres.** Pula główna; K ∈ {2, 3, 7, 14, 28}; ElecEner i GEFCom2012; scenariusze oracle i persystencja; doba D+1; te same doby co w analizie głównej i w profilu K (reguła kwietnia bez zmian).

**Porównania, wyłącznie opisowe.** Dla każdej pary (K, ψ): różnica MAE C9 minus C1 i C9 minus C3, przedział bootstrap (bloki 7 dób), test DM bez korekty Holma. Żadnego wyboru „najlepszego” ψ. Wynik podajemy jako krzywą MAE w funkcji ψ dla każdego K.

**Ochrona wyników zamrożonych.** Osobny skrypt czyta archiwum przebiegów tylko do odczytu i zapisuje wyniki do `results/<przebieg>/po_wynikach/regularyzacja/`. Sumy kontrolne SHA-256 wszystkich plików w `results/main/report`, `results/main/combined`, `results/gefcom2012/report` i `results/gefcom2012/combined` są liczone przed i po obliczeniu i wpisywane do sekcji 9.

**Ograniczenie, które ujawniamy.** Pomysł ściągania wag do równych wynika z zobaczenia, że średnia wygrywa. Analiza odpowiada więc na pytanie recenzenta, ale nie może zmienić tezy.
