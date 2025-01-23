import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import { useState, useEffect } from "react";
import "./PricePrediction.css";
function PricePrediction({ prediction, onDisable }) {
    const [isTransitioning, setIsTransitioning] = useState(false);

    useEffect(() => {
        const timer1 = setTimeout(() => {
            setIsTransitioning(true);
        }, 1000);

        return () => {
            clearTimeout(timer1);
        };
    }, []);

    const handleClose = () => {
        setIsTransitioning(false);
        setTimeout(() => {
            onDisable();
        }, 1000);
    };

    return (
        <div
            className={`animation-container ${
                isTransitioning ? "fade-out" : "visible"
            }`}
            onClick={() => handleClose()}
        >
            <DotLottieReact
                src="https://lottie.host/e6e9b70f-eef6-475e-9f44-e8b80d9ae850/44fWix1qYK.lottie"
                autoplay
            />
            <div
                className={`text-container ${
                    isTransitioning ? "fade-in" : "hidden"
                }`}
            >
                {prediction && <p>Predicted Price: {prediction} ¥</p>}
            </div>
        </div>
    );
}

export default PricePrediction;
