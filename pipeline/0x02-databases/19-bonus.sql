-- Creates a stored procedure AddBonus that adds a new correction for a student
CREATE PROCEDURE AddBonus (
    @user_id INT,
    @project_name NVARCHAR(50),
    @score INT
) AS BEGIN -- Check if the project already exists
IF NOT EXISTS(
    SELECT 1
    FROM projects
    WHERE name = @project_name
) BEGIN -- If the project doesn't exist, create a new project
INSERT INTO projects(name)
VALUES (@project_name);
END -- Get the project_id for the given project_name
DECLARE @project_id INT;
SET @project_id = (
        SELECT id
        FROM projects
        WHERE name = @project_name
    );
-- Insert the correction for the given user and project
INSERT INTO corrections(user_id, project_id, score)
VALUES (@user_id, @project_id, @score);
END
