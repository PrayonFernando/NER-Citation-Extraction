import React from "react";
import { TextField, Grid, Button } from "@mui/material";
import "./reviewStyle.css";
import Header from "../../components/Header/header";
import { Dropzone, FileMosaic } from "@dropzone-ui/react";
import { useNavigate, useLocation } from "react-router-dom"; // Import useNavigate
// import Review from "../Review/review";

function Review() {
  const navigate = useNavigate(); // Hook to programmatically navigate

  const location = useLocation();
  const files = location.state?.files;

  return (
    <Grid>
      {files &&
        files.map((file, index) => (
          <Grid key={index}>
            {/* Display file details */}
            File name: {file.name}
            {/* Render file preview if it's an image */}
            {file.type.startsWith("image/") && (
              <img
                src={URL.createObjectURL(file.file)}
                alt={file.name}
                style={{ maxWidth: "100%", height: "auto" }}
              />
            )}
          </Grid>
        ))}
    </Grid>
  );
}

export default Review;
