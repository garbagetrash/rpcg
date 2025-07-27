use macroquad::prelude::*;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

pub mod colormap;
use crate::colormap::*;


fn colormap(value: f64, colormap: &[[u8; 4]]) -> [u8; 4] {
    let idx = (127. * value + 127.) as usize;
    colormap[idx]
}

fn sink_edges(value: f64, xnorm: f64, ynorm: f64) -> f64 {
    let x = (xnorm - 0.5).abs();
    let y = (ynorm - 0.5).abs();
    value - 5.*(x*x+y*y)
}

#[macroquad::main("RPCG")]
async fn main() {
    let fbm = Fbm::<Perlin>::new(1337)
        .set_octaves(5)
        .set_frequency(0.005);
    let w = screen_width();
    let h = screen_height();
    let mut rgba = vec![];
    for row in 0..h as usize {
        for col in 0..w as usize {
            //let value = 127. * fbm.get([row as f64, col as f64]) + 127.;
            let value = fbm.get([col as f64, row as f64]);
            let value2 = sink_edges(value, col as f64 / w as f64, row as f64 / h as f64);
            //println!("{}, {}", value, value2);
            let value = colormap(value2, &cmap1);
            //println!("{}", value);
            rgba.push(value[0]);
            rgba.push(value[1]);
            rgba.push(value[2]);
            rgba.push(value[3]);
        }
    }

    let texture = Texture2D::from_rgba8(w as u16, h as u16, &rgba);

    loop {
        clear_background(BLACK);

        draw_texture(&texture, 0., 0., WHITE);
        next_frame().await
    }
}
