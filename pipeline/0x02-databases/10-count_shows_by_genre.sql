-- Lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each
SELECT g.name AS genre,
    COUNT(t.show_id) AS number_of_shows
FROM tv_genres AS g
    JOIN tv_show_genres AS t ON g.id = t.genre_id
GROUP BY g.id
ORDER BY number_of_shows DESC;