import { motion } from "framer-motion";

function PageFadeAnimation({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: "100vw" }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: "100vw" }}
      transition={{ duration: 0.75, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
}

export default PageFadeAnimation;