-- Creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student
CREATE PROCEDURE ComputeAverageScoreForUser (@user_id INT) AS BEGIN -- Compute the average score for the given user
DECLARE @avg_score FLOAT;
SET @avg_score = (
        SELECT AVG(score)
        FROM corrections
        WHERE user_id = @user_id
    );
-- Update the user's average_score in the users table
UPDATE users
SET average_score = @avg_score
WHERE id = @user_id;
END
