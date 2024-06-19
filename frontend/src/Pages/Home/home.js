import React, { useState } from "react";
import { TextField, Grid, Button, Box } from "@mui/material";
import { createStyles, makeStyles } from "@mui/styles"; // Updated import
import Header from "../../components/Header/header";
import { Dropzone, FileMosaic } from "@dropzone-ui/react";
import { useNavigate } from "react-router-dom";
import scourt from "../../assets/scourt.jpg";
import EntitiesDisplay from "../../components/Entities";

import MyDropzone from "../../components/MyDropzone";

const Home = () => {
  const [files, setFiles] = useState([]);
  const [entities, setEntities] = useState([]);

  const handleSubmit = () => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("file", file); // 'file' should match the expected field in your API
    });

    fetch("https://012c-34-16-159-207.ngrok-free.app/upload", {
      // Replace 'your-api-endpoint' with the actual endpoint path
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        // Assuming your response structure has a typo "entites" instead of "entities"
        const fetchedEntities = data.entites || []; // Adjust this line based on the correct property name
        setEntities(fetchedEntities);
        console.log("Success:", data);
        // Here you might want to navigate the user to another page or display the extracted entities
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  return (
    <div
      style={{
        // backgroundImage: `url(${scourt})`,
        backgroundRepeat: "no-repeat",
        backgroundSize: "contain",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
        }}
      >
        <Header />

        <h1>HOW JURISTMIND WORK?</h1>
        <h3>1. First Upload Your Document In the DropZone Area Below</h3>
        <h3>2. Then Hold For Few Seconds...</h3>
        <h3>
          3. Finally After The Document Is Processed You Can Now View The
          Citations
        </h3>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
          }}
        >
          <MyDropzone files={files} setFiles={setFiles} />
        </div>

        <Button
          variant="contained"
          onClick={handleSubmit}
          style={{
            backgroundColor: "#003399",
            marginTop: "20px",
            padding: "10px 20px",
            fontSize: "16px",
          }}
        >
          Submit
        </Button>
        {/* <p>Please upload the file in only .PDF format</p> */}
      </div>

      <div>
        <EntitiesDisplay entities={entities} />
      </div>
    </div>
  );
};

export default Home;
