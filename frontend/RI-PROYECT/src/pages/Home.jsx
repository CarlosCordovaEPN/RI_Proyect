import { useEffect } from "react";
import api from "../utils/axios";

const Home = () => {
  useEffect(() => {
    // Realiza una petición GET al backend
    api.get("/")
      .then((response) => {
        console.log("Respuesta del backend:", response.data);
      })
      .catch((error) => {
        console.error("Error al comunicarse con el backend:", error);
      });
  }, []);

  return <h1>¡Bienvenido a mi aplicación React!</h1>;
};

export default Home;
