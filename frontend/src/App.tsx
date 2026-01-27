import { Routes, Route } from 'react-router-dom'
import DisplayFrontPage from './front_page'

function App() {
  return (
        <Routes>
          <Route path="/" element={<DisplayFrontPage />} />
        </Routes>
  );
}

export default App;