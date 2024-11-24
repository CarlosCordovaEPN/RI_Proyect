import React from 'react';
import '../styles/styles.css';

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-800 to-black">
      {/* Fondo decorativo */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="floating-circle floating-circle-1"></div>
        <div className="floating-circle floating-circle-2"></div>
      </div>

      <div className="relative">
        <header className="text-center py-16 px-4">
          {/* Título con efecto 3D */}
          <h1 className="text-8xl font-extrabold text-3d-title">
          Reuters Retriever
          </h1>
          {/* Subtítulo mejorado */}
          <p className="text-lg text-subtitle mt-4">
            Donde las búsquedas encuentran su máxima precisión
          </p>
        </header>

        {/* Barra de búsqueda */}
        <div className="flex justify-center mt-8 px-4">
          <input
            type="text"
            placeholder="Escribe tu consulta aquí..."
            className="search-input"
          />
        </div>
      </div>
    </div>
  );
};

export default Home;
