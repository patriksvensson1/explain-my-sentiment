

function DisplayFrontPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex-1 flex flex-col items-center justify-center text-center px-6">
        <h1 className="text-3xl font-bold">Explain My Sentiment</h1>
      </header>

      <main className="flex-1 flex justify-center items-start pt-6">
        <label className="cursor-pointer inline-flex items-center gap-3 px-6 py-3 rounded-lg border">
          <span>Upload text file</span>
          <input type="file" className="hidden" />
        </label>
      </main>

      <div className="flex-1" />
    </div>
  );
}
export default DisplayFrontPage;
