import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";

function MyDropzone({ files, setFiles }) {
  const onDrop = useCallback((acceptedFiles) => {
    setFiles((prevFiles) => [...prevFiles, ...acceptedFiles]);
  }, []);

  const handleUpload = () => {
    if (files.length === 0) {
      console.log("No files selected.");
      return;
    }
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  // Function to handle file upload

  // Styles omitted for brevity...

  return (
    <div
      style={{
        // backgroundColor: "#003399",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        border: "2px solid #000", // Example border: solid, 2px, black
        borderRadius: "10px", // Optional: adds rounded corners
        padding: "20px", // Optional: adds some space inside the border
        boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)", // Optional: adds a shadow effect
        height: "250px",
        width: "500px",
        marginTop: "5%",
      }}
    >
      <div
        {...getRootProps()}
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          alignContent: "center",
        }}
      >
        <input {...getInputProps()} />
        <p>Please upload the file in only .PDF format</p>

        <button
          type="button"
          onClick={handleUpload}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            alignContent: "center",
          }}
        >
          Upload Files
        </button>
      </div>
    </div>
  );
}

export default MyDropzone;
