# Prognozowanie godzinowego zapotrzebowania na energię: symulacja operacyjna

Kod do eksperymentu z artykułu o wielomodelowym prognozowaniu zapotrzebowania
(wersja po recenzjach). Realizuje metodykę opisaną w dokumencie „Metodyka eksperymentu”:
prognoza wydawana o północy na 48 godzin, codzienna estymacja na przesuwanym oknie
84 dób, archiwum prognoz z każdej doby i porównanie sposobów łączenia modeli.

## Zasady, których pilnuje kod

1. **Żadnego zaglądania w przyszłość.** Model dostaje okno kończące się godzinę przed
   chwilą prognozy, a ramka przyszłości nie zawiera zapotrzebowania. Test
   `tests/test_no_leakage.py` podmienia całą przyszłość na śmieci i sprawdza, że prognozy
   są identyczne.
2. **Surowe dane są nienaruszalne.** Korekty (wykluczenia, poprawki wartości, flagi)
   są zadeklarowane w YAML zbioru i wykonywane w kodzie, z dziennikiem zmian.
3. **Wynik da się odtworzyć.** Przy każdym wywołaniu zapisują się (w `runs/`): pełna
   konfiguracja, treść plików identyfikacji, wersje bibliotek, skrót SHA-256 pliku danych.
4. **Przebieg da się wznowić i rozszerzyć.** Archiwum ma układ „model × doba”: każda para
   zapisuje się atomowo do osobnego pliku. Ponowne uruchomienie liczy tylko brakujące pary,
   więc dołożenie modelu (np. benchmarku na prośbę recenzenta) liczy tylko ten model.

## Struktura

```
configs/datasets/      opis zbioru: plik, kolumny, kalendarz, korekty
configs/experiments/   okresy, okno, modele, scenariusze temperatury, łączenie
src/prognozowanie/
  data/                loadery, korekty, kalendarz (święta, dni pomostowe, zmiana czasu)
  models/              modele jako wtyczki o wspólnym interfejsie fit/predict/fitted
  backtest/            pętla po dobach, scenariusze temperatury, archiwum
  combine/             sposoby łączenia C0-C5 z doborem par zgodnym z horyzontem
  evaluate/            MAE, RMSE, MASE, przekroje, test Diebolda-Mariano
  identify/            strojenie SVR i sieci (Optuna), wybór rzędów ARIMA
tests/                 testy automatyczne
data/raw/              surowe pliki danych (poza repozytorium)
results/               wyniki przebiegów (poza repozytorium)
```

## Instalacja

### Windows (zalecane: WSL2)

1. W PowerShellu: `wsl --install -d Ubuntu` (przy Docker Desktop WSL2 zwykle już jest;
   bez niego potrzebne uprawnienia administratora i restart). Dalsze kroki w terminalu Ubuntu.
2. Pakiety systemowe: `sudo apt-get install -y git curl gh libgomp1`
   (`libgomp1` potrzebuje LightGBM).
3. Zainstaluj uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. Sklonuj repozytorium i wejdź do katalogu.
5. `uv sync --frozen --extra gefcom` (środowisko `.venv` z wersjami z `uv.lock`). Wersję
   Pythona (3.12) wskazuje `.python-version`; uv pobierze ją sam. Na Pythonie 3.13 i 3.14
   statsforecast 2.0.1 nie ma gotowych pakietów i wymaga kompilatora C++.
6. Skopiuj surowy plik danych do `data/raw/ElecEner_long.xlsx`.
7. Sprawdzenie: `uv run prognozowanie check-data configs/datasets/elecener.yaml`
   oraz `uv run pytest`.

VS Code z rozszerzeniem „WSL” pozwala pracować na tym katalogu z poziomu Windowsa.

### Linux, chmura, Docker

`uv sync` jak wyżej albo `docker build -t prognozowanie .` i uruchamianie przez
`docker run -v $PWD/data:/app/data -v $PWD/results:/app/results prognozowanie <polecenie>`.

## Użycie

```bash
# 1. kontrola danych (zakres, korekty, kalendarz); opcjonalnie eksport przygotowanych danych
uv run prognozowanie check-data configs/datasets/elecener.yaml --export dane.xlsx

# 2. identyfikacja struktur na okresie identyfikacji (strojenie trwa kilka godzin)
uv run prognozowanie identify configs/experiments/main.yaml
#    następnie w configs/experiments/main.yaml:  identified_params: results/identify/main/base/identified.yaml

# 3. backtest, łączenie, raport (domyślnie dla obu scenariuszy potwierdzenia)
uv run prognozowanie run configs/experiments/main.yaml --jobs 8
uv run prognozowanie combine configs/experiments/main.yaml
uv run prognozowanie evaluate configs/experiments/main.yaml

# kontrola techniczna w trakcie (kompletność, awarie, zbieżność, czasy), bez miar dokładności
uv run prognozowanie status configs/experiments/main.yaml --start 2002-08-01 --end 2002-10-31

# próba techniczna na parametrach wstępnych (przed identyfikacją) wymaga jawnej zgody
uv run prognozowanie all configs/experiments/main.yaml --run-name proba --start 2002-07-10 --end 2002-07-20 --allow-provisional
```

Przebieg odmawia pracy na parametrach wstępnych (modele bez kompletnej identyfikacji: strojenie
i rzędy ARIMA) i na wyniku identyfikacji policzonym na innych danych.

### Kolejka zadań (długie obliczenia na własnym komputerze)

`scripts/kolejka.sh` wykonuje po kolei pliki `*.job` z katalogu kolejki (domyślnie
`/mnt/c/EnergyForecasting/kolejka`). Każdy plik to jedna linia: `update`, `test` albo
`prognozowanie <podpolecenie> configs/... [opcje]`. Inne polecenia, ścieżki bezwzględne, „..”
i znaki specjalne są odrzucane; nic nie przechodzi przez powłokę. Dziennik każdego zadania:
`kolejka/logs/`, sygnał życia: `kolejka/heartbeat`.

Wyniki: `results/<nazwa przebiegu>/` zawiera:

| Ścieżka | Zawartość |
|---|---|
| `forecasts/<model>/<doba>.parquet` | prognozy modelu ze wszystkich scenariuszy i losowań, prognozy pierwszego stopnia (`<model>__stage1`), wartości dopasowane z ostatnich `forecast.fitted_hours_saved` godzin okna (domyślnie 168) |
| `meta/<model>/<doba>.json` | specyfikacja modelu, status (także awaria), współczynniki, zbieżność, czas; zapisywany jako ostatni, więc oznacza parę jako policzoną |
| `config.yaml` | migawka wejść wspólnych: dane z korektami, SHA-256, horyzont, okno, scenariusze temperatury, ziarno |
| `runs/` | ślad każdego wywołania: pełna konfiguracja, środowisko, zakres prac |
| `combined/<scenariusz>__<losowanie>.parquet`, `weights/...` | łączenie, partiami według scenariusza i losowania (liczone od nowa z archiwum) |
| `report/` | ocena: tabele CSV, raport Markdown dla każdego scenariusza, wykresy |
| `status.md` | kontrola techniczna (`prognozowanie status`) |

Zasady wznawiania:

- zmiana wejść wspólnych (dane, korekty, okno, horyzont, scenariusze temperatury, ziarno) jest
  błędem: to inny eksperyment, potrzebna nowa nazwa przebiegu;
- dołożenie modelu liczy tylko ten model; usunięcie modelu z listy zostawia jego pliki,
  ale łączenie i raport biorą tylko modele z bieżącej konfiguracji;
- zmiana specyfikacji modelu jest błędem, dopóki nie wskaże się go jawnie:
  `--recompute NAZWA` (przelicza tylko ten model) albo nie nada mu się nowej nazwy;
- zmiana łączenia lub oceny nie wymaga przeliczania prognoz;
- awaria modelu w danej dobie jest wynikiem (zapisana w `meta/`, trafia do `failures.csv`),
  nie jest ponawiana przy wznowieniu; ponowienie: `--recompute NAZWA`.

### Raport

`results/<przebieg>/report/`: miary (MAE, RMSE, MASE) wg bloku horyzontu, typu dnia, miesiąca
i godziny horyzontu; test Diebolda-Mariano; przedziały bootstrapowe (bloki 7 dób); zbiór
modeli najlepszych (MCS, T_max); stabilność wag; lista awarii modeli; wykresy PNG z danymi CSV.
Hipotezy główne są zapisane w konfiguracji (`evaluation.families`, plan analizy:
`docs/Plan_analizy.md`): test DM wobec odniesienia rodziny, Holm w rodzinie z planowaną
liczebnością, kontrole konwencji (h = 2, pasmo 14 dób, bez HLN), reguła potwierdzenia
w scenariuszach temperatury zmierzonej i persystencji (`primary_confirmed.csv`). Analiza główna
liczy się na dobach wspólnych dla wariantów z rodzin hipotez; zestawienie wszystkich wariantów
na ich własnych dobach wspólnych; pokrycie każdego planowanego wariantu: `coverage_*.csv`.
Luki w datach są uwzględniane w wariancji HAC i w bootstrapie (bloki kolejnych dób kalendarza).
Analizy dodatkowe: profil K, pojedyncze okno 48 h, rozkład MAE średniej (2 min(P, N)),
wartości dopasowane kontra prognozy przy tej samej liczbie obserwacji, różnorodność błędów,
ablacja korekty ARIMA, próg opłacalności prognozy temperatury, koszt asymetryczny, usunięcie
jednej doby z historii wag, koszt obliczeń (dopasowanie, jedna prognoza, pętla scenariuszy).

Prognozy pierwszego stopnia modeli dwustopniowych są archiwizowane osobno (`<model>__stage1`),
więc ablacja „bez korekty ARIMA” nie wymaga osobnego przebiegu. Wariant C0 można policzyć
z różną liczbą godzin wartości dopasowanych (`combination.c0_fitted_hours`, np. `[48, 72, 168]`
daje warianty `C0`, `C0_H72`, `C0_H168`); 48 godzin to procedura artykułu. Dodatkowe pule łączenia
(`combination.extra_pools`: `plus_hong`, `plus_lgbm`) liczą się na tym samym archiwum. Benchmark LightGBM jest strojony w identyfikacji tą samą procedurą co SVR i sieć.

### Ponowna identyfikacja co kwartał (wariant dla recenzenta 3)

```bash
uv run prognozowanie identify configs/experiments/main.yaml --as-of 2002-11-01
```

Wynik trafia do `results/identify/main/as_of_2002-11-01/identified.yaml`. W kopii konfiguracji
eksperymentu zamiast `identified_params` podaje się `identified_schedule` (lista
`{valid_from, params}`); kod sprawdza, że każda identyfikacja kończy się przed `valid_from`.

### Walidacja zewnętrzna: GEFCom2012

```bash
uv run --extra gefcom python scripts/fetch_gefcom2012.py      # pobranie i kontrola wyrównania dat
uv run prognozowanie check-data configs/datasets/gefcom2012.yaml
uv run prognozowanie identify configs/experiments/gefcom2012.yaml
```

Używany jest najdłuższy ciągły fragment danych konkursowych (29.11.2006-29.06.2008, 579 dób),
suma 20 stref i średnia temperatura z 11 stacji (godzinowa, °F).

## Dodanie nowego zbioru danych

1. Skopiuj `configs/datasets/elecener.yaml` i zmień: ścieżkę, nazwy kolumn, kraj kalendarza
   (`country: US` itd.), strefę czasową, korekty.
2. Jeśli plik ma inny układ (np. wiele stref w kolumnach), dopisz funkcję w
   `src/prognozowanie/data/loaders.py` i zarejestruj ją w `LOADERS`.
3. Skopiuj eksperyment, wskaż nowy zbiór i dopasuj okresy.

## Dodanie nowego modelu

Klasa dziedzicząca po `Forecaster` z dekoratorem `@register("nazwa_typu")`, implementująca
`fit(hist)`, `predict(fut)` i ustawiająca `self._fitted`. Potem jedna pozycja w `models:`
konfiguracji eksperymentu.

## Uwagi metodyczne zapisane w kodzie

- SARIMA estymowana metodą warunkowej sumy kwadratów (CSS, Box i Jenkins). Dokładna
  wiarygodność (`method: ml`) przy okresie 168 h trwa 80-140 s i zajmuje ok. 5 GB RAM na
  dopasowanie; jest dostępna do kontroli (`scripts/compare_css_ml.py`). Prognozę z parametrów
  CSS liczy własna rekursja (`models/arima_css.py`), bo statsforecast po CSS nie ustawia stanu
  modelu; rekursja odtwarza reszty biblioteki z dokładnością 1e-13.
- Wagi C2 (nieujemne, suma 1) liczone dokładnie przez przegląd podzbiorów modeli (układy KKT),
  bez optymalizatora iteracyjnego, który potrafi po cichu zwrócić punkt startowy.
- Raport porównuje modele wyłącznie na parach (doba wydania, blok) wspólnych dla wszystkich
  modeli, więc każdy blok horyzontu ma jeden zbiór dób dla tabel, testu DM, bootstrapu i MCS.
- Awaria modelu w jednej dobie: tej doby nie da się połączyć, ale wagi kolejnych dób liczą się
  na dostępnej historii, jeśli brakuje najwyżej jednej doby na tydzień (K = 7: min. 6 dób,
  K = 28: min. 24; dla K = 2 i 3 brak tolerancji). Liczba dób użytych do wag jest zapisywana.
- Zbieżność optymalizatorów (wygładzanie, ARIMA reszt, SVR) zapisuje się w metadanych doby,
  podobnie jak informacja, że w horyzoncie jest święto nieobecne w oknie uczącym.
- Wznowienie przebiegu porównuje migawkę wszystkich wejść (eksperyment, zbiór z korektami,
  skrót pliku danych, treść plików identyfikacji); badanie Optuny ma nazwę zależną od wejść
  strojenia, więc zmiana danych lub przestrzeni zaczyna nowe badanie.
- Wygładzanie wykładnicze: stałe z optymalizatora zamiast siatki co 0,001. Wariant (trend,
  tłumienie) wybiera identyfikacja, tylko z inicjalizacją heurystyczną (inicjalizacja
  estymowana w większości dób nie osiąga zbieżności): trend addytywny ekstrapolowany na 48 h potrafi
  dać prognozy ujemne (GEFCom), a bez trendu optymalizator daje alfa = 1, czyli model bliski
  naiwnemu, jak w pierwotnej wersji artykułu.
- SVR: zmienna celu standaryzowana na oknie (jak w sieci), więc C i epsilon są w jednostkach
  odchylenia standardowego i ta sama przestrzeń strojenia działa dla zbiorów o różnej skali.
- Regresja z kodu odtwarza tabelę 1 pierwotnej wersji artykułu i wartości dopasowane ze
  STATISTICA (`tests/test_article_replication.py`, wymaga pliku Short w `data/raw/`).
- Sieć neuronowa: średnia z kilku uczeń o ustalonych ziarnach (wynik powtarzalny).
- Scenariusze temperatury nie wymagają ponownej estymacji: zmienia się tylko wejście prognozy.
