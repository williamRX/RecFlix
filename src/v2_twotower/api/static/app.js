document.addEventListener("DOMContentLoaded", () => {
    // Éléments de l'interface
    const tabLogin = document.getElementById("tab-login");
    const tabSignup = document.getElementById("tab-signup");
    const loginSection = document.getElementById("login-section");
    const signupSection = document.getElementById("signup-section");
    const authPanel = document.getElementById("auth-panel");
    const loaderPanel = document.getElementById("loader-panel");
    const recsPanel = document.getElementById("recommendations-panel");
    
    // Inputs & Boutons
    const userIdInput = document.getElementById("user-id-input");
    const selectGender = document.getElementById("select-gender");
    const selectAge = document.getElementById("select-age");
    const selectOccupation = document.getElementById("select-occupation");
    const zipInput = document.getElementById("zip-input");
    const searchMovieInput = document.getElementById("search-movie-input");
    const searchResultsDropdown = document.getElementById("search-results-dropdown");
    const selectedMoviesContainer = document.getElementById("selected-movies-container");
    const popularMoviesContainer = document.getElementById("popular-movies-container");
    
    const btnSubmitLogin = document.getElementById("btn-submit-login");
    const btnSubmitSignup = document.getElementById("btn-submit-signup");
    const btnLogout = document.getElementById("btn-logout");
    
    const recsTitle = document.getElementById("recs-title");
    const recsSubtitle = document.getElementById("recs-subtitle");
    const moviesGrid = document.getElementById("movies-grid");

    // État de l'application
    let selectedMovies = []; // Liste d'objets { movieId, title, rating }
    let searchTimeout = null;
    let username = "";
    let sessionRatings = []; // Liste de { movieId, rating }
    let currentProfilePayload = null; // Stocke la dernière requête de profil envoyée
    let currentTitle = "";
    let currentSubtitle = "";
    let activeModalMovieId = null; // Stocke l'ID du film actif dans la modal

    // Dictionnaires de libellés chargés du serveur pour reconstruire les descriptions custom
    let agesMetadata = {};
    let occupationsMetadata = {};

    // Initialisation
    loadFormMetadata();
    loadPopularMovies();

    // ==========================================================================
    // SELECTION DES ONGLETS (TABS)
    // ==========================================================================
    tabLogin.addEventListener("click", () => {
        tabLogin.classList.add("active");
        tabSignup.classList.remove("active");
        loginSection.classList.add("active");
        signupSection.classList.remove("active");
    });

    tabSignup.addEventListener("click", () => {
        tabSignup.classList.add("active");
        tabLogin.classList.remove("active");
        signupSection.classList.add("active");
        loginSection.classList.remove("active");
    });

    // ==========================================================================
    // REQUETES METADONNEES D'INITIALISATION
    // ==========================================================================
    async function loadFormMetadata() {
        try {
            // Âges
            const agesRes = await fetch("/api/v2/ages");
            const agesData = await agesRes.json();
            agesMetadata = agesData;
            selectAge.innerHTML = Object.entries(agesData).map(([val, label]) => 
                `<option value="${val}">${label}</option>`
            ).join("");

            // Professions
            const occRes = await fetch("/api/v2/occupations");
            const occData = await occRes.json();
            occupationsMetadata = occData;
            selectOccupation.innerHTML = Object.entries(occData).map(([val, label]) => 
                `<option value="${val}">${label}</option>`
            ).join("");
        } catch (e) {
            console.error("Erreur lors du chargement des métadonnées du formulaire :", e);
        }
    }

    async function loadPopularMovies() {
        try {
            const res = await fetch("/api/v2/movies/popular");
            const data = await res.json();
            
            popularMoviesContainer.innerHTML = data.results.map(movie => 
                `<span class="suggestion-tag" data-id="${movie.movieId}" data-title="${movie.title}">
                    + ${movie.title}
                 </span>`
            ).join("");

            // Ajouter le clic sur les suggestions populaires
            document.querySelectorAll(".suggestion-tag").forEach(tag => {
                tag.addEventListener("click", (e) => {
                    const movieId = parseInt(tag.getAttribute("data-id"));
                    const title = tag.getAttribute("data-title");
                    addSelectedMovie(movieId, title);
                });
            });
        } catch (e) {
            console.error("Erreur lors du chargement des films populaires :", e);
        }
    }

    // ==========================================================================
    // RECHERCHE DE FILMS (DEBOUNCE + DROPDOWN)
    // ==========================================================================
    searchMovieInput.addEventListener("input", (e) => {
        const query = e.target.value;
        
        clearTimeout(searchTimeout);
        if (!query || query.length < 2) {
            searchResultsDropdown.classList.remove("active");
            return;
        }

        // Debounce de 300ms pour éviter de surcharger le serveur à chaque lettre
        searchTimeout = setTimeout(async () => {
            try {
                const res = await fetch(`/api/v2/movies/search?q=${encodeURIComponent(query)}`);
                const data = await res.json();
                
                if (data.results.length === 0) {
                    searchResultsDropdown.innerHTML = `<div class="dropdown-item text-dim">Aucun film trouvé</div>`;
                } else {
                    searchResultsDropdown.innerHTML = data.results.map(movie => {
                        const imgHtml = movie.poster_path 
                            ? `<img class="dropdown-movie-thumbnail" src="${movie.poster_path}" alt="" loading="lazy">`
                            : `<div class="dropdown-movie-thumbnail-fallback">🎬</div>`;
                        return `
                            <div class="dropdown-item" data-id="${movie.movieId}" data-title="${movie.title}">
                                ${imgHtml}
                                <div class="dropdown-movie-info">
                                    <strong>${movie.title}</strong>
                                    <span class="dropdown-movie-genres">${movie.genres}</span>
                                </div>
                            </div>
                        `;
                    }).join("");

                    // Attacher le clic aux éléments du dropdown
                    document.querySelectorAll(".dropdown-item[data-id]").forEach(item => {
                        item.addEventListener("click", () => {
                            const movieId = parseInt(item.getAttribute("data-id"));
                            const title = item.getAttribute("data-title");
                            addSelectedMovie(movieId, title);
                            searchMovieInput.value = "";
                            searchResultsDropdown.classList.remove("active");
                        });
                    });
                }
                searchResultsDropdown.classList.add("active");
            } catch (error) {
                console.error("Erreur lors de la recherche de films :", error);
            }
        }, 300);
    });

    // Fermer le dropdown si on clique en dehors
    document.addEventListener("click", (e) => {
        if (!searchMovieInput.contains(e.target) && !searchResultsDropdown.contains(e.target)) {
            searchResultsDropdown.classList.remove("active");
        }
    });

    // ==========================================================================
    // GESTION DES FILMS SELECTIONNES (TAGS)
    // ==========================================================================
    function addSelectedMovie(movieId, title) {
        // Éviter les doublons
        if (selectedMovies.some(m => m.movieId === movieId)) return;

        selectedMovies.push({ movieId, title, rating: 5.0 }); // Par défaut : Bien (5.0)
        renderSelectedMovies();
    }

    function removeSelectedMovie(movieId) {
        selectedMovies = selectedMovies.filter(m => m.movieId !== movieId);
        renderSelectedMovies();
    }

    function renderSelectedMovies() {
        if (selectedMovies.length === 0) {
            selectedMoviesContainer.innerHTML = `<span class="text-dim font-size-small">Aucun film sélectionné</span>`;
            return;
        }

        selectedMoviesContainer.innerHTML = selectedMovies.map(movie => 
            `<div class="movie-tag" data-id="${movie.movieId}">
                <span>${movie.title}</span>
                <select class="movie-tag-rating" data-id="${movie.movieId}">
                    <option value="5.0" ${movie.rating === 5.0 ? 'selected' : ''}>🟢 Bien</option>
                    <option value="3.0" ${movie.rating === 3.0 ? 'selected' : ''}>🟡 Moyen</option>
                    <option value="1.0" ${movie.rating === 1.0 ? 'selected' : ''}>🔴 Nul</option>
                </select>
                <button class="movie-tag-close" data-id="${movie.movieId}">&times;</button>
             </div>`
        ).join("");

        // Attacher les écouteurs de changement de note
        document.querySelectorAll(".movie-tag-rating").forEach(select => {
            select.addEventListener("change", (e) => {
                const mid = parseInt(select.getAttribute("data-id"));
                const rating = parseFloat(e.target.value);
                const movie = selectedMovies.find(m => m.movieId === mid);
                if (movie) {
                    movie.rating = rating;
                }
            });
        });

        // Attacher l'évènement de suppression
        document.querySelectorAll(".movie-tag-close").forEach(btn => {
            btn.addEventListener("click", () => {
                const movieId = parseInt(btn.getAttribute("data-id"));
                removeSelectedMovie(movieId);
            });
        });
    }

    // ==========================================================================
    // ENVOI ET RECUPERATION DES RECOMMANDATIONS
    // ==========================================================================
    
    // Soumission Connexion (ID MovieLens ou Pseudo Persistant)
    btnSubmitLogin.addEventListener("click", async () => {
        const inputVal = userIdInput.value.trim();
        if (!inputVal) {
            alert("Veuillez saisir un identifiant MovieLens ou un pseudo.");
            return;
        }

        // Vérification si l'identifiant est numérique (Warm Start)
        const isNumericId = /^\d+$/.test(inputVal);
        
        if (isNumericId) {
            const userIdVal = parseInt(inputVal);
            if (userIdVal < 1 || userIdVal > 6040) {
                alert("Veuillez saisir un ID utilisateur MovieLens valide (compris entre 1 et 6040).");
                return;
            }
            
            username = `Utilisateur ${userIdVal}`;
            sessionRatings = []; // Réinitialiser le feedback de session
            
            currentProfilePayload = {
                user_id: userIdVal,
                ratings: sessionRatings,
                top_n: 12
            };
            currentTitle = `Utilisateur ID ${userIdVal}`;
            currentSubtitle = "Profil historique chargé";

            await fetchAndRenderRecommendations(currentProfilePayload, currentTitle, currentSubtitle);
        } else {
            // Reconnexion d'un compte personnalisé
            try {
                const res = await fetch(`/api/v2/users/${encodeURIComponent(inputVal)}`);
                if (!res.ok) {
                    if (res.status === 404) {
                        alert("Ce pseudonyme n'existe pas. Veuillez d'abord créer un profil.");
                    } else {
                        alert("Erreur lors de la connexion.");
                    }
                    return;
                }
                
                const userData = await res.json();
                
                username = userData.username;
                sessionRatings = userData.ratings || [];
                
                currentProfilePayload = {
                    username: username,
                    ratings: sessionRatings,
                    top_n: 12
                };
                
                const ageLabel = agesMetadata[userData.age] || `${userData.age} ans`;
                const occLabel = occupationsMetadata[userData.occupation] || "Autre";
                
                currentTitle = username;
                currentSubtitle = `Profil : ${userData.gender === 'M' ? 'Homme' : 'Femme'} | ${ageLabel} | ${occLabel}`;
                
                await fetchAndRenderRecommendations(currentProfilePayload, currentTitle, currentSubtitle);
            } catch (err) {
                alert("Impossible de se connecter : " + err.message);
            }
        }
    });

    // Soumission Création de Profil (Cold Start)
    btnSubmitSignup.addEventListener("click", async () => {
        const usernameVal = document.getElementById("username-input").value.trim();
        if (!usernameVal) {
            alert("Veuillez saisir un nom d'utilisateur.");
            return;
        }

        // Vérifier si le pseudo est déjà pris avant de continuer
        try {
            const checkRes = await fetch(`/api/v2/users/${encodeURIComponent(usernameVal)}`);
            if (checkRes.ok) {
                alert("Ce pseudonyme est déjà pris. Veuillez en choisir un autre.");
                return;
            }
        } catch (err) {
            console.error("Erreur lors de la vérification du pseudo :", err);
        }
        
        username = usernameVal;
        
        const gender = selectGender.value;
        const age = parseInt(selectAge.value);
        const occupation = parseInt(selectOccupation.value);
        const zipCode = zipInput.value.trim() || "00000";
        
        // Convertir la sélection initiale en notes de session
        sessionRatings = selectedMovies.map(m => ({ movieId: m.movieId, rating: m.rating }));

        currentProfilePayload = {
            username,
            gender,
            age,
            occupation,
            zip_code: zipCode,
            ratings: sessionRatings,
            top_n: 12
        };

        const ageLabel = selectAge.options[selectAge.selectedIndex].text;
        const occLabel = selectOccupation.options[selectOccupation.selectedIndex].text;
        
        currentTitle = username;
        currentSubtitle = `Profil : ${gender === 'M' ? 'Homme' : 'Femme'} | ${ageLabel} | ${occLabel}`;

        await fetchAndRenderRecommendations(currentProfilePayload, currentTitle, currentSubtitle);
    });

    // Appel API principal
    async function fetchAndRenderRecommendations(payload, title, subtitle) {
        const isAlreadyVisible = !recsPanel.classList.contains("hidden");
        
        if (!isAlreadyVisible) {
            authPanel.classList.add("hidden");
            loaderPanel.classList.remove("hidden");
            recsPanel.classList.add("hidden");
        }

        try {
            const res = await fetch("/api/v2/recommend", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || "Erreur de serveur");
            }

            const data = await res.json();

            // 2. Remplir le Dashboard
            recsTitle.innerText = title;
            recsSubtitle.innerText = subtitle;
            renderMovieCards(data.recommendations);

            // 3. Bascule d'écran vers les résultats
            if (!isAlreadyVisible) {
                loaderPanel.classList.add("hidden");
                recsPanel.classList.remove("hidden");
            }
        } catch (e) {
            alert("Erreur lors de la génération des recommandations : " + e.message);
            loaderPanel.classList.add("hidden");
            authPanel.classList.remove("hidden");
            recsPanel.classList.add("hidden");
        }
    }

    // ==========================================================================
    // RENDU DES CARTES DE FILMS
    // ==========================================================================
    function renderMovieCards(movies) {
        if (!movies || movies.length === 0) {
            moviesGrid.innerHTML = `<div class="full-width text-center text-gray">Aucun film recommandé.</div>`;
            return;
        }

        moviesGrid.innerHTML = movies.map(movie => {
            // Extraction du titre propre et de l'année (ex: "Toy Story (1995)" -> "Toy Story" & "1995")
            let title = movie.title;
            let year = "Inconnu";
            const yearMatch = movie.title.match(/\((\d{4})\)/);
            if (yearMatch) {
                year = yearMatch[1];
                title = movie.title.replace(/\s*\(\d{4}\)\s*/, "").trim();
            }

            // Déterminer le genre principal pour la classe de couleur de header
            const genresArray = movie.genres.split('|');
            const mainGenre = genresArray[0].toLowerCase().replace("'", "").replace(" ", "-");

            // Formatage du score sur 5
            const scorePercent = (movie.score / 5.0) * 100;
            const isHigh = movie.score >= 4.5;

            // Rendu de l'affiche TMDB
            const posterHtml = movie.poster_path 
                ? `<img class="movie-card-poster" src="${movie.poster_path}" alt="${title}" loading="lazy">`
                : '';

            // Note de feedback de l'utilisateur pour ce film en session
            const foundRating = sessionRatings.find(r => r.movieId === movie.movieId);
            const userRating = foundRating ? foundRating.rating : null;

            return `
                <div class="movie-card ${mainGenre}" data-id="${movie.movieId}">
                    <div class="movie-card-header">
                        ${posterHtml}
                        <span class="score-badge ${isHigh ? 'high' : ''}">${movie.score.toFixed(2)}/5</span>
                        <div class="score-progress-container">
                            <div class="score-progress-bar" style="width: ${scorePercent}%"></div>
                        </div>
                    </div>
                    <div class="movie-card-body">
                        <div class="movie-title-section">
                            <h4 class="movie-title" title="${movie.title}">${title}</h4>
                            <span class="movie-year">${year}</span>
                        </div>
                        <div class="movie-genres-container">
                            ${genresArray.map(g => `<span class="genre-pill">${g}</span>`).join("")}
                        </div>
                        <div class="card-feedback-section">
                            <span class="feedback-label">Votre avis :</span>
                            <div class="feedback-buttons">
                                <button class="feedback-btn nul ${userRating === 1.0 ? 'active' : ''}" data-id="${movie.movieId}" data-val="1.0" title="Nul (1/5)">🔴</button>
                                <button class="feedback-btn moyen ${userRating === 3.0 ? 'active' : ''}" data-id="${movie.movieId}" data-val="3.0" title="Moyen (3/5)">🟡</button>
                                <button class="feedback-btn bien ${userRating === 5.0 ? 'active' : ''}" data-id="${movie.movieId}" data-val="5.0" title="Bien (5/5)">🟢</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join("");

        // Attacher les évènements de clic aux cartes de films pour la modal
        document.querySelectorAll(".movie-card").forEach(card => {
            card.addEventListener("click", (e) => {
                // Ne pas ouvrir la modal si on clique sur un bouton de feedback
                if (e.target.closest(".feedback-btn")) return;
                
                const movieId = parseInt(card.getAttribute("data-id"));
                openMovieDetailsModal(movieId);
            });
        });

        // Attacher l'évaluation (feedback) en direct
        document.querySelectorAll(".feedback-btn").forEach(btn => {
            btn.addEventListener("click", async (e) => {
                e.stopPropagation(); // Évite le clic sur la carte et l'ouverture de la modal
                const movieId = parseInt(btn.getAttribute("data-id"));
                const rating = parseFloat(btn.getAttribute("data-val"));

                const index = sessionRatings.findIndex(r => r.movieId === movieId);
                if (index !== -1) {
                    if (sessionRatings[index].rating === rating) {
                        // Annule la note si on clique de nouveau sur le même avis
                        sessionRatings.splice(index, 1);
                    } else {
                        sessionRatings[index].rating = rating;
                    }
                } else {
                    sessionRatings.push({ movieId, rating });
                }

                // Relance immédiatement les recommandations adaptées au feedback
                if (currentProfilePayload) {
                    currentProfilePayload.ratings = sessionRatings;
                    await fetchAndRenderRecommendations(currentProfilePayload, currentTitle, currentSubtitle);
                }
            });
        });
    }

    // ==========================================================================
    // MODAL DE DETAILS DU FILM ET GESTION DES COMMENTAIRES
    // ==========================================================================
    const movieDetailModal = document.getElementById("movie-detail-modal");
    const closeModalBtn = document.getElementById("modal-close-btn");
    const commentForm = document.getElementById("comment-form");
    const commentInput = document.getElementById("comment-input");

    async function openMovieDetailsModal(movieId) {
        try {
            const res = await fetch(`/api/v2/movies/${movieId}`);
            if (!res.ok) throw new Error("Film introuvable");
            
            const movie = await res.json();
            activeModalMovieId = movieId;

            // Titre propre et année
            let title = movie.title;
            let year = "Inconnu";
            const yearMatch = movie.title.match(/\((\d{4})\)/);
            if (yearMatch) {
                year = yearMatch[1];
                title = movie.title.replace(/\s*\(\d{4}\)\s*/, "").trim();
            }

            document.getElementById("modal-movie-title").innerText = title;
            document.getElementById("modal-movie-year").innerText = year;
            
            const posterImg = document.getElementById("modal-movie-poster");
            if (movie.poster_path) {
                posterImg.src = movie.poster_path;
                posterImg.style.display = "block";
            } else {
                posterImg.style.display = "none";
            }

            const genresContainer = document.getElementById("modal-movie-genres");
            genresContainer.innerHTML = movie.genres.split('|').map(g => `<span class="genre-pill">${g}</span>`).join("");

            document.getElementById("modal-movie-director").innerText = movie.director;
            document.getElementById("modal-movie-cast").innerText = movie.top_3_cast.replace(/\|/g, ", ");
            document.getElementById("modal-movie-overview").innerText = movie.overview;

            // Rendu des commentaires
            renderCommentsList(movie.comments);

            // Affiche la modal
            movieDetailModal.classList.remove("hidden");
        } catch (error) {
            alert("Impossible de charger les détails du film : " + error.message);
        }
    }

    function renderCommentsList(comments) {
        const listContainer = document.getElementById("modal-comments-list");
        if (!comments || comments.length === 0) {
            listContainer.innerHTML = `<div class="text-dim text-center font-size-small" style="padding: 1.5rem 0;">Aucun commentaire pour ce film. Soyez le premier à commenter !</div>`;
            return;
        }

        listContainer.innerHTML = comments.map(c => {
            const date = new Date(c.created_at).toLocaleString("fr-FR", {
                day: "2-digit",
                month: "2-digit",
                year: "2-digit",
                hour: "2-digit",
                minute: "2-digit"
            });
            return `
                <div class="comment-item">
                    <div class="comment-author-row">
                        <span>${c.username}</span>
                        <span class="comment-date">${date}</span>
                    </div>
                    <p class="comment-text-content">${escapeHTML(c.comment_text)}</p>
                </div>
            `;
        }).join("");
    }

    function escapeHTML(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Publication d'un commentaire
    commentForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = commentInput.value.trim();
        if (!text || !activeModalMovieId) return;

        try {
            const res = await fetch(`/api/v2/movies/${activeModalMovieId}/comments`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    username: username || "Anonyme",
                    comment_text: text
                })
            });

            if (!res.ok) throw new Error("Erreur serveur lors de l'envoi du commentaire");

            commentInput.value = "";
            
            // Re-récupérer les commentaires mis à jour
            const movieRes = await fetch(`/api/v2/movies/${activeModalMovieId}`);
            const updatedMovie = await movieRes.json();
            renderCommentsList(updatedMovie.comments);
        } catch (error) {
            alert("Erreur lors de l'envoi : " + error.message);
        }
    });

    // Fermeture de la modal
    closeModalBtn.addEventListener("click", () => {
        movieDetailModal.classList.add("hidden");
        activeModalMovieId = null;
    });

    movieDetailModal.addEventListener("click", (e) => {
        if (e.target === movieDetailModal) {
            movieDetailModal.classList.add("hidden");
            activeModalMovieId = null;
        }
    });

    // ==========================================================================
    // DECONNEXION / RETOUR PORTAIL
    // ==========================================================================
    btnLogout.addEventListener("click", () => {
        recsPanel.classList.add("hidden");
        authPanel.classList.remove("hidden");
        // Réinitialiser les champs et l'état
        userIdInput.value = "";
        document.getElementById("username-input").value = "";
        selectedMovies = [];
        sessionRatings = [];
        username = "";
        currentProfilePayload = null;
        renderSelectedMovies();
    });
});
